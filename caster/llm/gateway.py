"""The LLM gateway. The fix for the local-model shard failures.

**Observed problem.** N shards point at one local server. The server generates
serially (or with small fixed parallelism). Request *i* waits behind *i−1*, so
its wall time grows with N while `LLM_TIMEOUT_SEC` is fixed. Shards die on
timeout, and because they share one server, they die *together*.

**Root cause.** One timeout is being asked to bound two different things: time
spent **waiting for a slot** and time spent **generating**. Only the second is a
property of the work. And there is no admission control anywhere — the queue
forms inside the backend, where the client cannot see it.

```
shards ──HTTP──▶ gateway ──▶ [ backend_0 … backend_{M-1} ]   local replicas
                        └──▶ cloud endpoint                  (optional, same pool)
```

The eight requirements, and where each lives:

1. **Speaks the OpenAI-compatible API** — `_Handler`. Shards set `LLM_BASE_URL`
   at the gateway and know nothing about M; `llm/client.py` is unchanged.
2. **Admission control** — `Gateway._slots`, a semaphore sized
   `Σ per-backend concurrency`. Requests queue **in the gateway**, not in a
   backend.
3. **Two timeouts, separately configured** — `queue_timeout` is willingness to
   wait for a slot (generous or unbounded; exceeding it is a *load signal*, not
   a failure, and is logged as `queue_timeout`, a distinct event).
   `service_timeout` bounds one generation once admitted and is derived from
   the work: `max_tokens / measured_tok_per_sec × margin`. Conflating these is
   the bug; they stay apart in config, logs and errors.
4. **Least-outstanding-requests routing**, not round-robin — generation times on
   this corpus vary by more than an order of magnitude.
5. **Health, restart, requeue** — `_route` evicts a failing backend, and
   `_Coalescer` holds the second caller of a cache key on the first's future, so
   N shards that miss the same key at the same moment pay **one** call. Without
   it the spend cap is honoured while the bill is N×. A requeued request is
   never billed twice because the upstream call is made once per key.
6. **Window verification at startup, per backend, blocking** — `verify_window`.
   Ollama picks the context window from available VRAM and **truncates** an
   over-long prompt instead of refusing it, and the served window reaches
   neither the cache key nor the record, so the same model id on two machines is
   two different instruments. This is a correctness property, not an ops nicety:
   long prompts are the steered arms', so silent truncation lands
   differentially on the comparison the experiment exists to make.
7. **One spend ledger, one global cap** — `SpendLedger`, rebuilt by summing the
   call log on first use so it survives a gateway restart.
8. **Graceful drain on SIGTERM** — stop admitting, finish in flight, report.

`caster serve` (`llm/serve.py`) owns backend *lifecycle*; this module owns
*routing*. The split matters: a shard never touches a backend's lifecycle, so a
shard exiting cannot tear down the backend the other shards are using — which
is exactly the failure where the first shard to finish stops the server.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import pathlib
import signal
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from caster import paths

DEFAULT_PORT = 11434 + 100  # 11534 — never the Ollama default; see serve.py


@dataclasses.dataclass
class Backend:
    """One upstream that speaks the OpenAI-compatible API."""

    url: str                       # e.g. http://127.0.0.1:11435/v1
    model: str
    concurrency: int = 1
    kind: str = "local"            # "local" | "cloud"
    context_tokens: int = 0        # 0 = unknown/unverified
    owned: bool = False            # did `caster serve` start this process?
    healthy: bool = True
    outstanding: int = 0
    failures: int = 0
    served: int = 0

    @property
    def name(self) -> str:
        return f"{self.kind}:{self.url}"


class SpendLedger:
    """One global ledger and one global cap.

    A per-process cap of `BUDGET_USD_CAP=25` across N shards is a real ceiling
    of 25N, and the arithmetic falls on the operator. The
    ledger is rebuilt by summing `usd` over the call log on first use, which is
    the recovery path that makes it survive a gateway restart.
    """

    def __init__(self, cap: float, log: pathlib.Path | None = None,
                 *, price_in: float = 0.0, price_out: float = 0.0) -> None:
        self.cap = float(cap)
        self.log = pathlib.Path(log or paths.CALLS_LOG)
        self.price_in = float(price_in)
        self.price_out = float(price_out)
        # A cap of 0 used to refuse the FIRST call — `spent (0.0) >= cap (0.0)`
        # is true before anything is spent — which is how a free local backend
        # ends up answering every request with 402 Payment Required.
        #
        # So `cap <= 0` means NO CAP, the way `LLM_CONTEXT_TOKENS=0` already
        # means "no window check". But only where it is safe: with a non-zero
        # price it is refused at construction instead, because the expensive
        # mistake is the other direction — an operator who writes 0 meaning
        # "spend nothing" and gets unlimited spending on a hosted model.
        self.uncapped = self.cap <= 0.0
        if self.uncapped and (self.price_in or self.price_out):
            raise ValueError(
                f"spend cap {self.cap} with a non-zero price "
                f"(in {self.price_in}/Mtok, out {self.price_out}/Mtok). "
                f"0 means 'no cap', which is not what you want on a priced "
                f"backend — set a real cap, or set both prices to 0 for a "
                f"local model."
            )
        self._lock = threading.Lock()
        self._spent = self._rebuild()

    def _rebuild(self) -> float:
        total = 0.0
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    total += float(json.loads(line).get("usd", 0.0))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return total

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    def check(self) -> None:
        if self.uncapped:
            return
        if self.spent >= self.cap:
            raise RuntimeError(
                f"global spend cap {self.cap} USD reached "
                f"(spent {self.spent:.4f}). Raise --spend-cap, or set it to 0 "
                f"for an unpriced local backend."
            )

    def charge(self, usd: float) -> None:
        with self._lock:
            self._spent += float(usd)

    def price(self, prompt_tokens: int, completion_tokens: int) -> float:
        return prompt_tokens / 1e6 * self.price_in + completion_tokens / 1e6 * self.price_out

    def record(self, row: dict[str, Any]) -> None:
        """ONE ledger row per upstream call, written by the gateway.

        This is the other half of the coalescing. Coalescing makes N shards
        that miss one key produce one upstream *call*; without the write moving
        here as well, each of the N clients still appends its own `usd` row to
        the shared log and the ledger shows N **charges**. the criterion is on
        the ledger, not on the upstream.
        """
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        self.charge(float(row.get("usd", 0.0)))


class _Coalescer:
    """One in-flight upstream call per cache key.

    The whole point of the nonce design (Invariant 1) is that arms with
    byte-identical prompts share one draw. N shards starting together therefore
    miss the same key at the same moment and all N call the model. An atomic
    rename handles the resulting *write race* and keeps the file correct, but it
    still pays N times. The gateway sees every request, so it is the one place
    that can hold the second caller on the first's future.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inflight: dict[str, threading.Event] = {}
        self._results: dict[str, tuple[int, bytes, dict[str, str]]] = {}
        self._seen: dict[str, float] = {}
        self.saved = 0

    def begin(self, key: str) -> tuple[bool, threading.Event]:
        """(is_leader, event). A follower waits on the event for the result."""
        with self._lock:
            ev = self._inflight.get(key)
            if ev is not None:
                self.saved += 1
                return False, ev
            ev = threading.Event()
            self._inflight[key] = ev
            return True, ev

    #: How long a finished result stays available for a late follower. Beyond
    #: this the follower would have hit its own service timeout anyway, and an
    #: unbounded result map is a slow memory leak in a process that runs for
    #: days.
    RESULT_TTL = 300.0

    def finish(self, key: str, result: tuple[int, bytes, dict[str, str]]) -> None:
        now = time.time()
        with self._lock:
            self._results[key] = result
            self._seen[key] = now
            if len(self._results) > 64:
                for stale in [k for k, t in self._seen.items()
                              if now - t > self.RESULT_TTL and k not in self._inflight]:
                    self._results.pop(stale, None)
                    self._seen.pop(stale, None)
            ev = self._inflight.pop(key, None)
        if ev is not None:
            ev.set()

    def abandon(self, key: str) -> None:
        """The leader failed. Wake every follower so each retries for itself."""
        with self._lock:
            ev = self._inflight.pop(key, None)
            self._results.pop(key, None)
            self._seen.pop(key, None)
        if ev is not None:
            ev.set()

    def take(self, key: str) -> tuple[int, bytes, dict[str, str]] | None:
        with self._lock:
            return self._results.get(key)


#: The trust store this module verifies against.
#:
#: `client.py` reaches a backend through the OpenAI SDK, which is httpx, which
#: trusts `certifi`. This module reaches one through `urllib`, which trusts
#: OpenSSL's compiled-in paths — empty on a macOS python.org build. While every
#: backend was a local Ollama over http the difference was invisible: none of
#: the three call sites below ever negotiated TLS. The first https backend makes
#: the gateway the only component in the codebase that trusts nothing, and the
#: symptom is a 502 whose body says CERTIFICATE_VERIFY_FAILED — which reads like
#: a network fault and is not one.
#:
#: `SSL_CERT_FILE` / `SSL_CERT_DIR` still win when set, because a TLS-inspecting
#: corporate proxy signs with a root certifi has never heard of and the operator
#: is the only one who can supply it.
_TLS_CONTEXT: ssl.SSLContext | None = None


def tls_context() -> ssl.SSLContext:
    global _TLS_CONTEXT
    if _TLS_CONTEXT is None:
        if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
            _TLS_CONTEXT = ssl.create_default_context()
        else:
            try:
                # `truststore` reads the OPERATING SYSTEM's trust store, which is
                # what the SDK half of this codebase already verifies against and
                # the only store that knows about a root an MDM pushed onto the
                # machine. certifi would not: a TLS-inspecting corporate proxy
                # signs with a root no public bundle ships.
                import truststore
                _TLS_CONTEXT = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            except ImportError:
                try:
                    import certifi
                    _TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
                except ImportError:
                    _TLS_CONTEXT = ssl.create_default_context()
    return _TLS_CONTEXT


def verify_window(base_url: str, model: str, expected: int, *,
                  timeout: float = 15.0) -> int:
    """Ask a backend what context window it is actually serving.

    Returns the served window. Raises when it differs from `expected`, because
    **refusing is the point**: a backend serving a smaller window than the
    protocol declares silently crops the longest prompts, which are the steered
    arms', so the truncation lands differentially on the comparison.

    Ollama exposes this on `/api/ps` (loaded models) and `/api/show`. A plain
    OpenAI-compatible endpoint exposes nothing, which is reported as 0 — unknown
    — and is only fatal when `expected` was set.
    """
    root = base_url.rstrip("/")
    for suffix in ("/v1", "/v1/"):
        if root.endswith(suffix):
            root = root[: -len(suffix)]
            break
    served = 0
    for path, extract in (
        ("/api/ps", lambda b: _dig_ps(b, model)),
        ("/api/show", lambda b: _dig_show(b)),
    ):
        try:
            req = urllib.request.Request(
                root + path,
                data=(json.dumps({"model": model}).encode() if path == "/api/show" else None),
                headers={"Content-Type": "application/json"},
                method="POST" if path == "/api/show" else "GET",
            )
            with urllib.request.urlopen(req, timeout=timeout,
                                        context=tls_context()) as resp:
                served = extract(json.loads(resp.read().decode())) or served
        except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError):
            continue
        if served:
            break
    if expected and served and served != expected:
        raise RuntimeError(
            f"{base_url} serves a {served}-token context for {model!r}, but the "
            f"protocol declares {expected}. Refusing: the window reaches no "
            f"cache key and no record, so two machines would be two instruments "
            f"sharing one name, and the longest prompts — the steered arms' — "
            f"would be cropped differentially."
        )
    return served


def _dig_ps(blob: dict[str, Any], model: str) -> int:
    for entry in blob.get("models", []) or []:
        if entry.get("model", "").startswith(model) or entry.get("name", "").startswith(model):
            ctx = entry.get("context_length") or entry.get("context_size")
            if ctx:
                return int(ctx)
    return 0


def _dig_show(blob: dict[str, Any]) -> int:
    info = blob.get("model_info") or {}
    for k, v in info.items():
        if k.endswith(".context_length") and v:
            return int(v)
    ctx = (blob.get("parameters") or "")
    for line in str(ctx).splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            return int(parts[1])
    return 0


def auto_replicas(
    *,
    vram_free_bytes: int,
    model_bytes: int,
    shards: int,
    kv_headroom: float = 0.35,
    shards_per_replica: int = 4,
    max_replicas: int = 8,
) -> tuple[int, dict[str, int]]:
    """VRAM-first, shard-count-second. Shards do not create capacity.

        by_memory = floor(vram_free / (model_bytes × (1 + kv_headroom)))
        by_demand = ceil(shards / shards_per_replica)
        replicas  = max(1, min(by_memory, by_demand, max_replicas))

    So 4 shards → 1 replica and 8 shards → 2, but *capped* by what actually
    fits. A machine that can hold one replica gets one, and the extra shards
    wait in the gateway queue — which is the correct behaviour and no longer
    fatal. Both bounds are returned so the chosen value and its reasons go into
    the run manifest.
    """
    denom = max(1, int(model_bytes * (1.0 + kv_headroom)))
    by_memory = max(0, vram_free_bytes // denom)
    by_demand = -(-max(1, shards) // max(1, shards_per_replica))  # ceil
    chosen = max(1, min(by_memory or 1, by_demand, max_replicas))
    return chosen, {"by_memory": int(by_memory), "by_demand": int(by_demand),
                    "max_replicas": int(max_replicas)}


class Gateway:
    """Admission control, routing, coalescing, one ledger, one cap."""

    def __init__(
        self,
        backends: list[Backend],
        *,
        queue_timeout: float | None = None,
        service_timeout: float = 1800.0,
        spend_cap: float = 100.0,
        calls_log: pathlib.Path | None = None,
        log_path: pathlib.Path | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
        respawn: Any = None,
    ) -> None:
        if not backends:
            raise ValueError("a gateway needs at least one backend")
        self.backends = backends
        self.queue_timeout = queue_timeout     # None = unbounded (the default)
        self.service_timeout = float(service_timeout)
        self.ledger = SpendLedger(
            spend_cap, calls_log,
            price_in=(price_in if price_in is not None
                      else float(os.environ.get("PRICE_IN_PER_MTOK", 0))),
            price_out=(price_out if price_out is not None
                       else float(os.environ.get("PRICE_OUT_PER_MTOK", 0))),
        )
        #: "evict one that fails; RESPAWN it". `caster serve gateway`
        #: passes `llm.serve.start`; without it `rehealth` can only wait for a
        #: backend to come back on its own.
        self.respawn = respawn
        self.coalescer = _Coalescer()
        self._slots = threading.Semaphore(sum(max(1, b.concurrency) for b in backends))
        self._lock = threading.Lock()
        self._draining = threading.Event()
        self._sessions = 0            # refcount; see the teardown rule
        self.log_path = pathlib.Path(log_path or (paths.LOGS_DIR / "gateway.jsonl"))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats = {"admitted": 0, "queue_timeout": 0, "service_timeout": 0,
                      "coalesced": 0, "upstream_errors": 0, "evicted": 0}

    # -- bookkeeping ---------------------------------------------------------

    def event(self, kind: str, **fields: Any) -> None:
        """One structured line per notable event.

        `queue_timeout` and `service_timeout` are **distinct, separately
        labelled** events, because the acceptance criterion is on the service
        timeout alone: queueing 8 shards behind one replica is *expected* to
        exceed any finite queue timeout, and that is a load signal, not a
        failure.
        """
        self.stats[kind] = self.stats.get(kind, 0) + 1
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.time(), "event": kind, **fields}) + "\n")

    def register(self) -> None:
        with self._lock:
            self._sessions += 1

    def unregister(self) -> None:
        with self._lock:
            self._sessions = max(0, self._sessions - 1)

    @property
    def sessions(self) -> int:
        with self._lock:
            return self._sessions

    # -- routing -------------------------------------------------------------

    def _pick(self) -> Backend | None:
        """Least-outstanding-requests among healthy backends.

        Round-robin would be wrong here: generation times on this corpus vary by
        more than an order of magnitude, so the next backend in sequence is not
        the next free one.

        the hybrid rule rides on top: **cloud is the overflow.** Admit to a
        local slot when one is free, else to cloud if under the spend cap, else
        queue. So a local backend with a free slot always outranks a cloud one,
        whatever the outstanding counts say.
        """
        with self._lock:
            healthy = [b for b in self.backends if b.healthy]
            if not healthy:
                return None
            free_local = [b for b in healthy
                          if b.kind == "local" and b.outstanding < max(1, b.concurrency)]
            pool = free_local or healthy
            chosen = min(pool, key=lambda b: (b.outstanding, b.served, b.url))
            chosen.outstanding += 1
            return chosen

    def _release(self, backend: Backend, ok: bool) -> None:
        with self._lock:
            backend.outstanding = max(0, backend.outstanding - 1)
            if ok:
                backend.served += 1
                backend.failures = 0
            else:
                backend.failures += 1
                if backend.failures >= 3:
                    backend.healthy = False
                    self.stats["evicted"] += 1

    def rehealth(self) -> None:
        """Probe evicted backends, bring back the ones that answer, respawn the rest.

        A requeued request is never billed twice because the upstream call is
        made once per cache key and the ledger row is written by whoever
        actually completed it — and a retry produces the same key anyway, since
        a request is deterministic in (prompt, model, temperature, nonce).
        """
        for b in self.backends:
            if b.healthy:
                continue
            try:
                with urllib.request.urlopen(b.url.rstrip("/") + "/models", timeout=5,
                                            context=tls_context()):
                    pass
            except urllib.error.HTTPError as exc:
                # The probe deliberately carries no Authorization, so a hosted
                # backend answers 401. That is a server that is UP. Letting it
                # fall through to URLError (whose subclass it is) left a cloud
                # backend evicted for the life of the process: `respawn` is
                # local-only, so the `else` below just skipped it forever and
                # every later call was a 502 nobody could explain.
                if exc.code not in (401, 403):
                    continue
            except (urllib.error.URLError, OSError, TimeoutError):
                if self.respawn is not None and b.kind == "local":
                    try:
                        self.respawn(b)
                        self.event("respawned", backend=b.name)
                    except Exception as exc:  # noqa: BLE001 - reported, not fatal
                        self.event("respawn_failed", backend=b.name, error=str(exc))
                        continue
                else:
                    continue
            with self._lock:
                b.healthy = True
                b.failures = 0

    # -- the call ------------------------------------------------------------

    def forward(self, body: bytes, headers: dict[str, str]) -> tuple[int, bytes, dict[str, str]]:
        """Admit, route, call upstream, return. Coalesced by cache key."""
        if self._draining.is_set():
            return 503, b'{"error":"gateway draining"}', {"Content-Type": "application/json"}
        try:
            self.ledger.check()
        except RuntimeError as exc:
            return 402, json.dumps({"error": str(exc)}).encode(), {"Content-Type": "application/json"}

        # Case-insensitive: HTTP header names are, and a proxy that normalises
        # them would otherwise silently disable coalescing — one call becomes N
        # and the bill goes back to N× with nothing in the log to say so.
        key = next((v for k, v in headers.items()
                    if k.lower() == "x-caster-cache-key" and v), None) \
            or hashlib.sha256(body).hexdigest()
        leader, ev = self.coalescer.begin(key)
        if not leader:
            self.stats["coalesced"] += 1
            # The follower waits on the leader's generation, bounded by the same
            # service timeout: it is waiting for the same work.
            if not ev.wait(timeout=self.service_timeout):
                return 504, b'{"error":"coalesced wait exceeded service_timeout"}', \
                       {"Content-Type": "application/json"}
            got = self.coalescer.take(key)
            if got is None:
                return 502, b'{"error":"leader failed; retry"}', {"Content-Type": "application/json"}
            return got

        try:
            if not self._slots.acquire(timeout=self.queue_timeout):
                self.event("queue_timeout", key=key[:12], waited=self.queue_timeout)
                return 429, json.dumps({
                    "error": "queue_timeout",
                    "detail": "no backend slot became free within queue_timeout. "
                              "This is a LOAD signal, not a failure: raise "
                              "queue_timeout (or leave it unbounded) or add "
                              "replicas.",
                }).encode(), {"Content-Type": "application/json"}
            try:
                return self._forward_once(key, body, headers)
            finally:
                self._slots.release()
        finally:
            if self.coalescer.take(key) is None:
                self.coalescer.abandon(key)

    def _forward_once(self, key: str, body: bytes,
                      headers: dict[str, str]) -> tuple[int, bytes, dict[str, str]]:
        # Case-insensitive, for exactly the reason the cache-key lookup in
        # `forward` already is — and this one was missed. HTTP header names are
        # case-insensitive, and httpx, which the OpenAI SDK speaks through,
        # sends them lowercased. A `.get("Authorization")` therefore missed
        # EVERY call made by the client half of this codebase and substituted
        # the placeholder, so the upstream answered 401 "Incorrect API key
        # provided: unused" — while the identical request from curl, which
        # sends the capitalised name, succeeded. The placeholder stays for the
        # genuinely keyless case: a local Ollama wants no credential.
        auth = next((v for k, v in headers.items()
                     if k.lower() == "authorization" and v), "Bearer unused")
        attempts = 0
        last: tuple[int, bytes, dict[str, str]] = (
            502, b'{"error":"no healthy backend"}', {"Content-Type": "application/json"})
        while attempts < max(2, len(self.backends)):
            attempts += 1
            backend = self._pick()
            if backend is None:
                self.rehealth()
                backend = self._pick()
            if backend is None:
                return last
            self.stats["admitted"] += 1
            req = urllib.request.Request(
                backend.url.rstrip("/") + "/chat/completions",
                data=body, method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": auth,
                    "X-Caster-Cache-Key": key,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=self.service_timeout,
                                            context=tls_context()) as resp:
                    payload = resp.read()
                    status = resp.status
                self._release(backend, ok=True)
                payload = self._settle(key, payload, backend)
                result = (status, payload, {"Content-Type": "application/json",
                                            "X-Caster-Backend": backend.name})
                self.coalescer.finish(key, result)
                return result
            except (TimeoutError, socket.timeout):
                self.event("service_timeout", key=key[:12], backend=backend.name,
                           limit=self.service_timeout)
                self._release(backend, ok=False)
                last = (504, json.dumps({
                    "error": "service_timeout",
                    "detail": f"one generation exceeded service_timeout="
                              f"{self.service_timeout}s on {backend.name}. Size it "
                              f"from max_tokens / measured_tok_per_sec × margin.",
                }).encode(), {"Content-Type": "application/json"})
            except urllib.error.HTTPError as exc:
                payload = exc.read()
                self._release(backend, ok=exc.code < 500)
                if exc.code < 500:
                    # A 4xx is OUR request being wrong — a bad argument, a model
                    # that does not exist. Retrying it on another backend just
                    # produces the same 4xx, so it is returned.
                    result = (exc.code, payload, {"Content-Type": "application/json"})
                    self.coalescer.finish(key, result)
                    return result
                self.event("upstream_errors", key=key[:12], backend=backend.name, code=exc.code)
                last = (exc.code, payload, {"Content-Type": "application/json"})
            except (urllib.error.URLError, OSError) as exc:
                # A read timeout arrives wrapped in URLError. The two timeouts
                # must stay DISTINCT, separately-labelled events, so it is
                # unwrapped here rather than filed as a generic failure.
                # A read timeout surfaces as TimeoutError; a CONNECT timeout
                # surfaces as URLError(socket.timeout()). Both are unwrapped
                # here, so the log distinguishes "waiting for a slot" from
                # "generating" wherever the timeout came from.
                inner = getattr(exc, "reason", None)
                if isinstance(exc, (TimeoutError, socket.timeout)) or \
                        isinstance(inner, (TimeoutError, socket.timeout)) or \
                        "timed out" in str(exc).lower():
                    self.event("service_timeout", key=key[:12], backend=backend.name,
                               limit=self.service_timeout)
                    self._release(backend, ok=False)
                    last = (504, json.dumps({
                        "error": "service_timeout",
                        "detail": f"one generation exceeded service_timeout="
                                  f"{self.service_timeout}s on {backend.name}.",
                    }).encode(), {"Content-Type": "application/json"})
                    continue
                self.event("upstream_errors", key=key[:12], backend=backend.name,
                           error=str(exc))
                self._release(backend, ok=False)
                last = (502, json.dumps({"error": f"backend unreachable: {exc}"}).encode(),
                        {"Content-Type": "application/json"})
        return last

    def _settle(self, key: str, payload: bytes, backend: Backend) -> bytes:
        """Charge the one global ledger, write the one ledger row, and stamp the
        reply so the client knows not to write a second one.

        The stamp travels as a top-level field in the JSON body rather than as a
        header, because the OpenAI SDK hands the caller a parsed model and does
        not surface response headers — which is why the client's old
        `getattr(r, "_caster_backend", ...)` could never resolve.
        """
        try:
            blob = json.loads(payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return payload
        usage = blob.get("usage") or {}
        tin = int(usage.get("prompt_tokens") or 0)
        tout = int(usage.get("completion_tokens") or 0)
        usd = self.ledger.price(tin, tout)
        self.ledger.record({
            "cache_key": key, "model": blob.get("model"),
            "in": tin, "out": tout,
            "finish_reason": (blob.get("choices") or [{}])[0].get("finish_reason"),
            "usd": round(usd, 6), "backend": backend.name, "at": time.time(),
            "via": "gateway",
        })
        blob["_caster_gateway"] = True
        blob["_caster_backend"] = backend.name
        blob["_caster_usd"] = round(usd, 6)
        return json.dumps(blob).encode()

    # -- lifecycle -----------------------------------------------------------

    def drain(self) -> None:
        """Stop admitting, let in-flight finish, report.

        Teardown waits on the REFCOUNT as well as on in-flight work: a shard
        that has registered but is between calls still owns the pool, and
        tearing it down there is the same failure as the first shard to finish
        stopping the backend the others are using.
        """
        self._draining.set()
        deadline = time.time() + self.service_timeout
        while time.time() < deadline:
            with self._lock:
                busy = sum(b.outstanding for b in self.backends) + self._sessions
            if busy == 0:
                break
            time.sleep(0.2)
        self.event("drained", stats=dict(self.stats), spent=self.ledger.spent)


class _Handler(BaseHTTPRequestHandler):
    gateway: Gateway

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return  # the gateway writes its own structured log

    def _send(self, status: int, payload: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_DELETE(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/session", "/v1/session"):
            self.gateway.unregister()
            self._send(200, json.dumps({"sessions": self.gateway.sessions}).encode(),
                       {"Content-Type": "application/json"})
            return
        self._send(404, b'{"error":"not found"}', {"Content-Type": "application/json"})

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/healthz", "/v1/healthz"):
            body = json.dumps({
                "ok": any(b.healthy for b in self.gateway.backends),
                "backends": [dataclasses.asdict(b) for b in self.gateway.backends],
                "stats": self.gateway.stats,
                "spent_usd": round(self.gateway.ledger.spent, 6),
                "cap_usd": self.gateway.ledger.cap,
                "coalesced": self.gateway.coalescer.saved,
                "sessions": self.gateway.sessions,
            }).encode()
            self._send(200, body, {"Content-Type": "application/json"})
            return
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            models = sorted({b.model for b in self.gateway.backends})
            body = json.dumps({"object": "list",
                               "data": [{"id": m, "object": "model"} for m in models]}).encode()
            self._send(200, body, {"Content-Type": "application/json"})
            return
        self._send(404, b'{"error":"not found"}', {"Content-Type": "application/json"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/session", "/v1/session"):
            # Teardown is driven by a REFCOUNT, not by shard exit. A
            # shard registers when it starts talking to the gateway and
            # unregisters when it stops; the gateway holds the pool up while any
            # session is registered and drains on SIGTERM.
            self.gateway.register()
            self._send(200, json.dumps({"sessions": self.gateway.sessions}).encode(),
                       {"Content-Type": "application/json"})
            return
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._send(404, b'{"error":"not found"}', {"Content-Type": "application/json"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length)
        headers = {k: v for k, v in self.headers.items()}
        status, payload, out_headers = self.gateway.forward(body, headers)
        self._send(status, payload, out_headers)


def serve_forever(gateway: Gateway, host: str = "127.0.0.1",
                  port: int = DEFAULT_PORT) -> None:
    """Run the gateway until SIGTERM/SIGINT, then drain."""
    handler = type("_BoundHandler", (_Handler,), {"gateway": gateway})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True

    def _shutdown(signum: int, frame: Any) -> None:  # noqa: ARG001
        threading.Thread(target=lambda: (gateway.drain(), httpd.shutdown()),
                         daemon=True).start()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _shutdown)
        except (ValueError, OSError):  # pragma: no cover - not the main thread
            pass

    print(f"[gateway] listening on http://{host}:{port}/v1  "
          f"backends={len(gateway.backends)} "
          f"slots={sum(max(1, b.concurrency) for b in gateway.backends)} "
          f"queue_timeout={gateway.queue_timeout} "
          f"service_timeout={gateway.service_timeout}", flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        gateway.drain()


def service_timeout_for(max_tokens: int, tok_per_sec: float, margin: float = 2.0) -> float:
    """Bound one generation from the work, not from a wall-clock guess.

    `max_tokens / measured_tok_per_sec × margin`, floored so a tiny draw against
    a cold backend is not killed by arithmetic.
    """
    if tok_per_sec <= 0:
        return float(os.environ.get("LLM_TIMEOUT_SEC", 1800))
    return max(60.0, (max_tokens / tok_per_sec) * margin)
