"""One OpenAI-compatible call path, plus the cross-run response cache.

The wire format is OpenAI's chat-completions. Any endpoint that speaks it —
`api.openai.com`, a local Ollama, the CASTER gateway — differs only in
`LLM_BASE_URL`, `LLM_API_KEY` and `MODEL`. There is one code path, not a
provider switch, so a smoke run against a cheap endpoint exercises the same code
the real run does.

The one exception is the reasoning (o-series) profile: it takes
`max_completion_tokens` instead of `max_tokens`, refuses any `temperature` but
its own default, and spends hidden reasoning tokens that are billed as output.
`REASONING_EFFORT` then becomes part of the protocol in the same way `MODEL` is
— it changes the proposal distribution, so it is in the cache key, on the
record and in the cell key, or two efforts are two instruments sharing one name.

**The cache is cross-run** (Invariant 8) and keyed on a caller-supplied nonce,
not on the prompt alone. Without a nonce two calls with byte-identical prompts
replay each other — which quietly destroys the experiment, because an unsteered
arm builds the *same* prompt every round of every seed. The nonce is
deterministic (`keys.draw_nonce`), so re-running a cell still replays from cache
for free, while rounds that are meant to be independent draws really are.

Three proposal failures are a **taxonomy**, and each entry was paid for:

| class               | what the loop does                                      |
|---------------------|---------------------------------------------------------|
| `BackendUnreachable`| record a spent round, count consecutive; at 5, stop cleanly |
| `ContextOverflow`   | record a spent round and continue — **a deterministic poison pill if it escapes** |
| `TruncatedResponse` | record a spent round and continue, **without touching the ledger** |

`BackendUnreachable` exists because the SDK retries inside **one** call and then
raises: left uncaught, a single connection blip kills a shard — and under a
fleet, every shard at once, because they share one server. `ContextOverflow`
must be caught because the prompt is a pure function of (task, seed, round,
policy): an uncaught overflow aborts at the identical round on every re-run, and
under atomic cells the cell is then never written and resume retries it forever.
`TruncatedResponse` must never reach the ledger: a reply the output budget cut
short is a harness failure, and letting one become evidence pollutes exactly the
mechanism under study.

Two fields the record needs, and that are easy to leave out:

* **`finish_reason`** — "ran out of output budget" and "answered in the wrong
  format" both reach the parser as a missing closing fence, and neither is
  visible in the text alone.
* **a charged/uncharged bit** — a harness-failure round is charged against the
  budget deliberately (leaving it free lets a task whose proposer keeps
  truncating draw until a cap), so which rounds were charged has to be on the
  row, or `success@B` is a curve of an unknown B.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import time
from typing import Any

from caster import paths
from caster.errors import CacheMiss
from caster.keys import cache_key

#: Same figures the output budget is sized with (`steer.template`). Duplicated
#: here rather than imported because the window check must not depend on the
#: prompt builder.
CHARS_PER_TOKEN = 3.5
BUDGET_MARGIN = 1.15


class BudgetExceeded(RuntimeError):
    """The spend cap was reached. The gateway owns the global one."""


class BackendUnreachable(RuntimeError):
    """The model server did not answer, after every retry. Not a timeout."""


class ContextOverflow(RuntimeError):
    """Prompt + `max_tokens` would not fit the server's context window."""


class TruncatedResponse(RuntimeError):
    """The reply carried no *closed* ```python fence.

    Two causes land here and the name only names the first: a reply cut off
    mid-block, and a reply that answered in prose and stopped — which is what a
    weaker model does to a format instruction it ignores. `finish_reason` tells
    them apart: `length` is a real truncation and the output budget is too
    small; `stop` is a model that simply never emitted a block.

    The tempting fallback — return the raw reply, prose and opening fence
    included, as though it were a patch — always produces a candidate that
    fails, so a harness defect enters memory as evidence and pollutes exactly
    the mechanism under study. Raising instead lets the loop log the round as
    spent-but-inconclusive
    and leave the ledger untouched.
    """


@dataclasses.dataclass
class CallMeta:
    """What one call cost. Goes on the round record; `cache_key` is the join key.

    The draw nonce is **not** a join key — one nonce maps to several ledger rows
    the moment arms diverge (Invariant 1's corollary).
    """

    cache_key: str = ""
    model: str = ""
    max_tokens: int = 0
    reasoning_effort: str | None = None
    cached: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    llm_sec: float | None = None
    usd: float = 0.0
    backend: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_REASONING_PREFIXES = ("o1", "o3", "o4", "o5")


def is_reasoning(model: str) -> bool:
    """Matched on the leading token of the id, so `o4-mini`,
    `o4-mini-2025-04-16` and `o3-mini` route the same way while `gpt-4o-mini` —
    whose id starts with `gpt` — does not."""
    return bool(model) and model.split("-", 1)[0] in _REASONING_PREFIXES


class LLMClient:
    """One client, one cache, one ledger. Built once per process.

    Built once rather than per call: against a local server every construction
    is a fresh connection pool, and the SDK's retry budget is per-client.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        cache_dir: pathlib.Path | None = None,
        calls_log: pathlib.Path | None = None,
        context_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int | None = None,
        price_in: float | None = None,
        price_out: float | None = None,
        spend_cap: float | None = None,
    ) -> None:
        self.model = model if model is not None else os.environ.get("MODEL", "")
        # Empty base url means the SDK's own default. An endpoint that ignores
        # the key still needs one present: the SDK refuses to construct a client
        # without it, hence the placeholder.
        self.base_url = base_url if base_url is not None else (os.environ.get("LLM_BASE_URL", "") or None)
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "") or "unused"
        self.temperature = float(
            temperature if temperature is not None else os.environ.get("TEMPERATURE", 1.0)
        )
        self.reasoning_effort = (
            reasoning_effort if reasoning_effort is not None
            else os.environ.get("REASONING_EFFORT", "")
        )
        self.cache_dir = pathlib.Path(cache_dir or paths.CACHE_DIR)
        self.calls_log = pathlib.Path(calls_log or paths.CALLS_LOG)
        # Total context the server is ACTUALLY serving, or 0 to disable.
        #
        # This exists because a server is free to truncate rather than refuse.
        # The window is server-side, the OpenAI-compatible endpoint has no field
        # to raise it, and a server that crops instead of erroring returns a
        # plausible answer to a question that was never asked. Worse, the window
        # reaches neither the cache key nor the report, so the same model id
        # under a smaller window is a different instrument and nothing says so.
        # Hosted endpoints error rather than crop — this check is what makes
        # that a property of the code and not of the provider.
        self.context_tokens = int(
            context_tokens if context_tokens is not None
            else os.environ.get("LLM_CONTEXT_TOKENS", 0)
        )
        # The SDK's default is 600s. A local 7B at ~16 tok/s needs ~1000s to
        # fill a 16000-token ceiling, so the default would abort a legitimate
        # call and record it as a harness failure.
        self.timeout = float(timeout if timeout is not None else os.environ.get("LLM_TIMEOUT_SEC", 1800))
        self.max_retries = int(max_retries if max_retries is not None else os.environ.get("LLM_MAX_RETRIES", 5))
        self.price_in = float(price_in if price_in is not None else os.environ.get("PRICE_IN_PER_MTOK", 0))
        self.price_out = float(price_out if price_out is not None else os.environ.get("PRICE_OUT_PER_MTOK", 0))
        self.spend_cap = float(spend_cap if spend_cap is not None else os.environ.get("BUDGET_USD_CAP", 100))
        # `cap <= 0` means "no cap", the same as `gateway.SpendLedger`. Without
        # this the two halves disagreed: an unpriced local backend passed the
        # gateway and then died on `0.0 >= 0.0` here, on the FIRST call, before
        # any spending existed to cap. Refused at construction when a price is
        # set, because the expensive mistake is the other direction — an
        # operator who writes 0 meaning "spend nothing" and is given unlimited
        # spending on a hosted model.
        self.uncapped = self.spend_cap <= 0.0
        if self.uncapped and (self.price_in or self.price_out):
            raise ValueError(
                f"spend cap {self.spend_cap} with a non-zero price "
                f"(in {self.price_in}/Mtok, out {self.price_out}/Mtok). "
                f"0 means 'no cap', which is not what you want on a priced "
                f"backend — set a real cap, or set both prices to 0 for an "
                f"unpriced local backend."
            )
        self._client: Any = None
        self._spent: float | None = None
        #: Calls that actually reached a model, counted here rather than read
        #: back off the call ledger. The gateway owns that ledger now and may be
        #: running under a different RUN_DIR, which would make `caster smoke
        #: --live` report a real call as a cache hit.
        self.live_calls = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_log.parent.mkdir(parents=True, exist_ok=True)

    # ── the cache ───────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> pathlib.Path:
        # Two levels of fan-out: a flat cache of 200k+ blobs makes `ls` and
        # every directory operation on it quadratic on some filesystems.
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if not path.is_file():
            # Flat layout, so an installed artifact bundle still resolves.
            flat = self.cache_dir / f"{key}.json"
            if not flat.is_file():
                return None
            path = flat
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def _write_cache(self, key: str, blob: dict[str, Any]) -> None:
        """Atomic: temp in the same directory, then `os.replace`.

        Two shards running in parallel against one cache will both miss the same
        key and both write it. `os.replace` is atomic on POSIX, so a reader can
        only ever see the whole blob or no file at all — never the half-written
        JSON a plain `write_text` can expose. (Correctness, not cost: the
        gateway's cache-key coalescing is what stops paying N times.)
        """
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(blob), encoding="utf-8")
        os.replace(tmp, path)

    # ── the spend ledger ────────────────────────────────────────────────────

    def spent(self) -> float:
        """USD charged so far, per the call ledger.

        Read from the file once, then kept in memory and advanced by `complete`.
        Re-reading on every call made this O(n²) over a sweep that logs tens of
        thousands of them. The consequence is that spend by a *concurrently*
        running driver is invisible here — which is exactly why the gateway owns
        the single global ledger and the cap. A per-process cap of
        `BUDGET_USD_CAP=25` across N shards is a real ceiling of 25N.
        """
        if self._spent is None:
            total = 0.0
            if self.calls_log.exists():
                for line in self.calls_log.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        total += float(json.loads(line).get("usd", 0.0))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue
            self._spent = total
        return self._spent

    def _log_call(self, row: dict[str, Any]) -> None:
        with self.calls_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ── the gateway session (the refcount) ───────────────────────────────

    def register_session(self) -> bool:
        """Tell the gateway a shard is working, so it does not tear the pool down.

        Teardown is driven by a refcount, not by shard exit. A shard between
        calls still owns the pool, and tearing it down there is the same failure
        as the first shard to finish stopping the backend the others are using.
        Best effort: a plain endpoint has no /session, and the run must not
        depend on one.
        """
        return self._session("POST")

    def unregister_session(self) -> bool:
        return self._session("DELETE")

    def _session(self, method: str) -> bool:
        if not self.base_url:
            return False
        import urllib.error
        import urllib.request

        url = self.base_url.rstrip("/").removesuffix("/v1") + "/session"
        try:
            req = urllib.request.Request(url, method=method, data=b"" if method == "POST" else None)
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            return False

    # ── the call ────────────────────────────────────────────────────────────

    def _sdk(self) -> Any:
        if self._client is None:
            import openai

            self._client = openai.OpenAI(
                base_url=self.base_url, api_key=self.api_key,
                timeout=self.timeout, max_retries=self.max_retries,
            )
        return self._client

    def check_window(self, prompt: str, max_tokens: int) -> None:
        """The per-call window check. Survives the gateway.

        The gateway's startup probe verifies what a backend *serves*; it says
        nothing about whether *this* prompt fits. Both are needed and they fail
        at different times.
        """
        if not self.context_tokens:
            return
        estimate = math.ceil(len(prompt) / CHARS_PER_TOKEN * BUDGET_MARGIN)
        if estimate + max_tokens > self.context_tokens:
            raise ContextOverflow(
                f"~{estimate} prompt + {max_tokens} output tokens exceeds the "
                f"{self.context_tokens}-token window. Raise the server's own "
                f"window rather than letting it truncate the prompt."
            )

    def complete(
        self,
        prompt: str,
        *,
        nonce: str,
        max_tokens: int = 1024,
        model: str | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        meta: CallMeta | None = None,
        from_cache_only: bool = False,
    ) -> str:
        """One model call, cached on disk.

        `meta` is filled with what the call cost — `cache_key`, token counts,
        seconds, `finish_reason` — so the caller can put them on its own record.
        A cache **hit** fills it too, so a replayed cell reports the same
        numbers a live one did.
        """
        model = model or self.model
        temperature = self.temperature if temperature is None else float(temperature)
        effort = self.reasoning_effort if reasoning_effort is None else reasoning_effort
        if not model:
            raise ValueError("no model configured — set MODEL or pass model=...")
        if effort and not is_reasoning(model):
            raise ValueError(
                f"REASONING_EFFORT={effort!r} was set but {model!r} is not a "
                f"reasoning model. The effort would enter the cache key and the "
                f"cell key while changing nothing about the call, so every arm "
                f"would look like a distinct protocol for no reason."
            )

        key = cache_key(prompt, model, temperature, max_tokens, nonce, effort)
        if meta is not None:
            meta.cache_key = key
            meta.model = model
            meta.max_tokens = max_tokens
            meta.reasoning_effort = effort or None

        blob = self.cached(key)
        if blob is not None:
            if meta is not None:
                meta.cached = True
                meta.prompt_tokens = blob.get("in")
                meta.completion_tokens = blob.get("out")
                meta.reasoning_tokens = blob.get("reasoning_out")
                meta.finish_reason = blob.get("finish_reason")
                meta.llm_sec = blob.get("sec")
                meta.backend = blob.get("backend")
            return blob.get("text", "")

        if from_cache_only:
            # NOT BackendUnreachable. That class is caught by the loop, charged
            # as a spent round and fed to the circuit breaker, so five misses
            # would end the cell as a "backend outage" — exactly the silent
            # degradation a replay-only run exists to make loud. A miss means phase
            # 1 has not drawn this round, which is a build failure.
            raise CacheMiss(
                f"--from-cache-only: cache key {key[:12]} is a miss for nonce "
                f"{nonce!r}. Phase 1 has not drawn this round. A silent model "
                f"call here is what phase separation exists to prevent."
            )

        if not self.uncapped and self.spent() >= self.spend_cap:
            raise BudgetExceeded(
                f"cap {self.spend_cap} USD reached (spent {self.spent():.4f}). "
                f"Raise --spend-cap, or set it to 0 for an unpriced local "
                f"backend."
            )

        self.check_window(prompt, max_tokens)

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if is_reasoning(model):
            kwargs["max_completion_tokens"] = max_tokens
            if effort:
                kwargs["reasoning_effort"] = effort
            # `temperature` is deliberately not sent: the reasoning profile
            # accepts only its own default and errors on anything else. It stays
            # in the cache key, where it records what the protocol asked for.
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        # The gateway coalesces concurrent misses of one cache key onto a single
        # upstream call, and a cache key is a function of the nonce,
        # which is not part of the OpenAI wire format. So it travels as a
        # header. A plain endpoint ignores unknown headers, so the same code
        # path serves both.
        kwargs["extra_headers"] = {
            "X-Caster-Cache-Key": key,
            "X-Caster-Nonce": nonce,
        }

        t0 = time.time()
        self.live_calls += 1
        try:
            r = self._sdk().chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - narrowed immediately below
            # Matched on module+name rather than isinstance, deliberately: an
            # `import openai` inside this handler raises ModuleNotFoundError of
            # its own on a host without the package, which would MASK whatever
            # actually went wrong. Found by running exactly that.
            mod, name = (type(exc).__module__ or ""), type(exc).__name__
            if not (mod.split(".")[0] == "openai"
                    and name in ("APIConnectionError", "APITimeoutError")):
                raise
            raise BackendUnreachable(f"{name}: {exc}") from exc

        choice = r.choices[0]
        # `or ""`: content is None, not "", when the model emits no text at all.
        # The parser runs a regex over this and would raise TypeError instead of
        # the TruncatedResponse the loop knows how to log.
        text = choice.message.content or ""
        tin = int(getattr(r.usage, "prompt_tokens", 0) or 0)
        tout = int(getattr(r.usage, "completion_tokens", 0) or 0)
        details = getattr(r.usage, "completion_tokens_details", None)
        treason = int(getattr(details, "reasoning_tokens", 0) or 0)
        sec = round(time.time() - t0, 2)
        # The gateway stamps the reply with a top-level `_caster_gateway` field,
        # which the SDK surfaces through `model_extra` (response headers are not
        # exposed on the parsed model). When it is there, the gateway has
        # already charged its one global ledger and written the one row — N
        # coalesced shards writing their own would show N charges for one call
        #.
        extra = getattr(r, "model_extra", None) or {}
        via_gateway = bool(extra.get("_caster_gateway"))
        backend = extra.get("_caster_backend") or self.base_url or "default"
        usd = (float(extra.get("_caster_usd") or 0.0) if via_gateway
               else tin / 1e6 * self.price_in + tout / 1e6 * self.price_out)

        self._write_cache(key, {
            "text": text, "model": model, "temperature": temperature, "nonce": nonce,
            "in": tin, "out": tout, "reasoning_out": treason,
            "finish_reason": choice.finish_reason, "sec": sec, "backend": backend,
        })
        if not via_gateway:
            self._log_call({
                "model": model, "temperature": temperature, "nonce": nonce,
                "reasoning_effort": effort or None, "cache_key": key,
                "in": tin, "out": tout, "reasoning_out": treason,
                "finish_reason": choice.finish_reason,
                "usd": round(usd, 6), "sec": sec, "backend": backend,
                "at": time.time(),
            })
        if meta is not None:
            meta.cached = False
            meta.prompt_tokens = tin
            meta.completion_tokens = tout
            meta.reasoning_tokens = treason
            meta.finish_reason = choice.finish_reason
            meta.llm_sec = sec
            meta.usd = round(usd, 6)
            meta.backend = backend
        self._spent = self.spent() + usd
        return text
