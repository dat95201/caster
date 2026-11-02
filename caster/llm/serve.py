"""Local backend lifecycle, owned by the gateway and nobody else.

A local backend is a first-class venue here, not a thing an operator wires up
by hand. Setting `LLM_BASE_URL` at a server you started yourself skips every
check below — the served window above all — so the lifecycle is owned, and it is
owned by **one** process rather than by each shard.

```
serve start   --port P --model M --context C     start, then verify
serve verify  --port P                           ask the server its served window; hard-stop on mismatch
serve unload  --port P                           drop the weights, leave the process up
serve stop    --port P                           unload, then stop the listener WE started
```

**Why `--keep-serving` is the default, not a flag.** If each shard started and
stopped the server, the first shard to finish would tear down the backend the
others were still using. The fix is an ownership rule, not a per-shard flag:

* the **gateway** starts the replicas and is the only process that may stop them;
* a **shard never touches a backend's lifecycle** — it exits, the pool is unaffected;
* the gateway tears down **only what it started**. A replica it *adopted* is
  `unload`ed at most, never killed: on a shared machine the process is a few MB
  and the resident weights are gigabytes;
* `--keep-serving` survives as a *gateway* flag meaning "do not even unload on
  exit", for chaining several runs against one warm pool. Start-up on a 7B is
  minutes; paying it between phases is pure waste;
* teardown is driven by a **refcount**, not by shard exit.

When a listener is found on a replica's port at start-up:

| it is                                        | the gateway does                          |
|----------------------------------------------|-------------------------------------------|
| ours, previous run, right model + window      | adopt, refcount++                          |
| ours, wrong model or window                   | refuse and say so — never silently re-use a different instrument |
| not ours (e.g. the desktop app on the default port) | adopt read-only, never unload, never stop |

Replicas use their **own port range** (from 11435 upward), never the Ollama
default 11434, so a user's own server is never the one a run tears down.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from caster import paths
from caster.llm.gateway import verify_window

#: Never 11434. That is Ollama's default and very likely the user's own server.
FIRST_REPLICA_PORT = 11435
DESKTOP_DEFAULT_PORT = 11434

#: Where this machine records which listeners CASTER started. Not under RUN_DIR:
#: ownership is a property of the machine, and a second run with a different
#: RUN_DIR must still know it did not start the process it finds.
OWNERSHIP_FILE = paths.ROOT / ".caster" / "serve.json"


@dataclasses.dataclass
class Replica:
    port: int
    model: str
    context: int
    pid: int | None = None
    owned: bool = False
    adopted: bool = False
    read_only: bool = False     # someone else's server: never unload, never stop

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    def root_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _load_ownership() -> dict[str, Any]:
    if not OWNERSHIP_FILE.is_file():
        return {"replicas": {}}
    try:
        return json.loads(OWNERSHIP_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"replicas": {}}


def _save_ownership(doc: dict[str, Any]) -> None:
    OWNERSHIP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = OWNERSHIP_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(OWNERSHIP_FILE)


def port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _loaded_model(port: int, timeout: float = 5.0) -> str | None:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/ps", timeout=timeout) as r:
            blob = json.loads(r.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, OSError, TimeoutError):
        return None
    models = blob.get("models") or []
    if not models:
        return ""      # listening, nothing resident
    return models[0].get("model") or models[0].get("name") or ""


def start(port: int, model: str, context: int, *, num_parallel: int = 1,
          wait_s: float = 120.0) -> Replica:
    """Start a replica on `port`, then verify its served window. Blocking.

    Adoption is decided here, by the table in the module docstring, and the
    decision is persisted so a later `stop` knows what it may kill.
    """
    if port == DESKTOP_DEFAULT_PORT:
        raise ValueError(
            f"port {DESKTOP_DEFAULT_PORT} is Ollama's default and very likely the "
            f"user's own server. Replicas start at {FIRST_REPLICA_PORT} so a run "
            f"can never tear down a server it did not start."
        )

    own = _load_ownership()
    record = own["replicas"].get(str(port))

    if port_in_use(port):
        resident = _loaded_model(port)
        ours = bool(record)
        if not ours:
            # Someone else's server. Adopt READ-ONLY: usable, never unloaded,
            # never stopped.
            rep = Replica(port=port, model=model, context=context,
                          adopted=True, read_only=True)
            verify(rep)
            return rep
        if resident and model and not resident.startswith(model.split(":")[0]):
            raise RuntimeError(
                f"a CASTER replica is already listening on {port} serving "
                f"{resident!r}, but this run wants {model!r}. Refusing to "
                f"silently re-use a different instrument. Run "
                f"`caster serve stop --port {port}` first."
            )
        rep = Replica(port=port, model=model, context=context,
                      pid=record.get("pid"), owned=True, adopted=True)
        verify(rep)          # the window check is the other half of "same instrument"
        return rep

    exe = shutil.which("ollama")
    if exe is None:
        raise RuntimeError(
            "no `ollama` on PATH and nothing is listening on "
            f"{port}. Start an OpenAI-compatible server yourself and point "
            f"--gateway-url at it, or install ollama."
        )
    env = dict(os.environ)
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    env["OLLAMA_NUM_PARALLEL"] = str(max(1, num_parallel))
    if context:
        env["OLLAMA_CONTEXT_LENGTH"] = str(context)
    log_dir = paths.LOGS_DIR / "serve"
    log_dir.mkdir(parents=True, exist_ok=True)
    log = (log_dir / f"ollama-{port}.log").open("a", encoding="utf-8")
    proc = subprocess.Popen([exe, "serve"], env=env, stdout=log, stderr=log,
                            start_new_session=True)

    deadline = time.time() + wait_s
    while time.time() < deadline:
        if port_in_use(port):
            break
        if proc.poll() is not None:
            raise RuntimeError(
                f"ollama serve on {port} exited with {proc.returncode}; see "
                f"{log_dir / f'ollama-{port}.log'}"
            )
        time.sleep(0.5)
    else:
        raise TimeoutError(f"ollama serve did not come up on {port} within {wait_s}s")

    own["replicas"][str(port)] = {"pid": proc.pid, "model": model,
                                  "context": context, "started_at": time.time()}
    _save_ownership(own)

    rep = Replica(port=port, model=model, context=context, pid=proc.pid, owned=True)
    if model:
        _pull_and_load(rep)
    verify(rep)
    return rep


def _pull_and_load(rep: Replica, timeout: float = 1800.0) -> None:
    """Make the weights resident, so the window probe has something to report."""
    body = json.dumps({"model": rep.model,
                       "messages": [{"role": "user", "content": "ok"}],
                       "max_tokens": 1, "stream": False}).encode()
    req = urllib.request.Request(rep.base_url + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer unused"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except (urllib.error.URLError, OSError, TimeoutError):
        # A failed warm-up is not fatal here: `verify` is what decides whether
        # this backend may serve, and it reports a far more useful message.
        pass


def verify(rep: Replica) -> int:
    """Ask the server what window it is serving. Hard-stop on a mismatch."""
    served = verify_window(rep.base_url, rep.model, rep.context)
    if rep.context and not served:
        raise RuntimeError(
            f"{rep.base_url} would not report its served context window, and the "
            f"protocol declares {rep.context}. A backend that crops instead of "
            f"erroring returns a plausible answer to a question that was never "
            f"asked, and the window reaches no cache key and no record. "
            f"Refusing to serve it."
        )
    return served


def unload(rep: Replica) -> bool:
    """Drop the weights, leave the process up. Never on a read-only adoption."""
    if rep.read_only:
        return False
    body = json.dumps({"model": rep.model, "keep_alive": 0}).encode()
    req = urllib.request.Request(rep.root_url + "/api/generate", data=body,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60):
            return True
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


def stop(rep: Replica) -> bool:
    """Unload, then stop the listener **we** started. Adopted ⇒ unload only."""
    unload(rep)
    if rep.read_only or not rep.owned or rep.adopted:
        return False
    own = _load_ownership()
    record = own["replicas"].get(str(rep.port))
    pid = rep.pid or (record or {}).get("pid")
    if not pid:
        return False
    try:
        os.kill(int(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, ValueError):
        pass
    own["replicas"].pop(str(rep.port), None)
    _save_ownership(own)
    return True


def describe(port: int, *, model: str = "", context: int = 0) -> Replica:
    """What this machine knows about the listener on `port`.

    `caster serve stop|unload --port P` must not act on a listener CASTER never
    started, and a bare `Replica(owned=False)` does not say that: `stop()` calls
    `unload()` before it checks ownership, so the weights of someone else's
    server would be dropped. The ownership record is the only authority on
    "ours", so it is consulted here and the flags come back right.
    """
    own = _load_ownership()
    record = own["replicas"].get(str(port))
    if record is None:
        return Replica(port=port, model=model or "", context=context,
                       read_only=True, owned=False)
    return Replica(port=port, model=model or record.get("model", ""),
                   context=context or int(record.get("context", 0) or 0),
                   pid=record.get("pid"), owned=True)


def plan_ports(n: int, first: int = FIRST_REPLICA_PORT) -> list[int]:
    """`n` consecutive replica ports, skipping the desktop default."""
    out: list[int] = []
    port = first
    while len(out) < n:
        if port != DESKTOP_DEFAULT_PORT:
            out.append(port)
        port += 1
    return out
