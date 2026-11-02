"""The gateway and the local-backend lifecycle."""
from __future__ import annotations

import json
import threading
import time
import urllib.request

import pytest

from caster.llm.gateway import (
    Backend,
    Gateway,
    SpendLedger,
    _Coalescer,
    auto_replicas,
    serve_forever,
    service_timeout_for,
)
from caster.llm import serve


# ── admission control and the two timeouts ───────────────────────────────────


def test_queue_timeout_and_service_timeout_are_different_events():
    """One timeout was being asked to bound two different things.

    Only generation is a property of the work. Queueing 8 shards behind one
    replica is EXPECTED to exceed any finite queue timeout, and that is a load
    signal, not a failure — so the acceptance criterion is on the service
    timeout alone and the two must appear as distinct labelled events.
    """
    gw = Gateway([Backend(url="http://127.0.0.1:1/v1", model="m", concurrency=1)],
                 queue_timeout=0.05, service_timeout=0.05)
    # Hold the only slot so the next request cannot be admitted.
    assert gw._slots.acquire(timeout=1)
    status, payload, _ = gw.forward(b"{}", {"X-Caster-Cache-Key": "k1"})
    gw._slots.release()
    assert status == 429
    blob = json.loads(payload)
    assert blob["error"] == "queue_timeout"
    assert "LOAD signal" in blob["detail"]
    assert gw.stats["queue_timeout"] == 1
    assert gw.stats["service_timeout"] == 0


def test_service_timeout_is_sized_from_the_work():
    """`max_tokens / measured_tok_per_sec × margin` — the number that was being
    blown by queueing."""
    assert service_timeout_for(16_000, 16.0, margin=2.0) == pytest.approx(2000.0)
    assert service_timeout_for(10, 1000.0) == 60.0        # floored
    assert service_timeout_for(10, 0) > 0                 # unmeasured: fall back


def test_n_concurrent_misses_of_one_key_make_one_upstream_call():
    """Without coalescing the spend cap is honoured while the bill is N×.

    An atomic rename handles the resulting write race and keeps the file
    correct, but it still pays N times. The gateway sees every request, so it is
    the one place that can hold the second caller on the first's future.
    """
    c = _Coalescer()
    leader, _ = c.begin("k")
    assert leader
    followers = []
    for _ in range(4):
        is_leader, ev = c.begin("k")
        followers.append((is_leader, ev))
    assert not any(f[0] for f in followers)
    assert c.saved == 4
    c.finish("k", (200, b"body", {}))
    for _, ev in followers:
        assert ev.wait(1)
    assert c.take("k")[1] == b"body"


def test_an_abandoned_key_wakes_every_follower():
    c = _Coalescer()
    c.begin("k")
    _, ev = c.begin("k")
    c.abandon("k")
    assert ev.wait(1)
    assert c.take("k") is None


def test_routing_is_least_outstanding_not_round_robin():
    """Generation times on this corpus vary by more than an order of magnitude."""
    a = Backend(url="http://a/v1", model="m")
    b = Backend(url="http://b/v1", model="m")
    gw = Gateway([a, b])
    first = gw._pick()
    second = gw._pick()
    assert {first.url, second.url} == {"http://a/v1", "http://b/v1"}
    gw._release(second, ok=True)
    assert gw._pick().url == second.url      # the free one, not the next one round


def test_an_unhealthy_backend_is_evicted_and_can_come_back():
    b = Backend(url="http://b/v1", model="m")
    gw = Gateway([b])
    for _ in range(3):
        gw._pick()
        gw._release(b, ok=False)
    assert not b.healthy and gw.stats["evicted"] == 1
    assert gw._pick() is None


def test_one_global_spend_ledger_survives_a_restart(tmp_path):
    """A per-process cap of C across N shards is a real ceiling of C·N, and
    the arithmetic falls on the operator."""
    log = tmp_path / "calls.jsonl"
    log.write_text("\n".join(json.dumps({"usd": 0.4}) for _ in range(3)) + "\n",
                   encoding="utf-8")
    ledger = SpendLedger(cap=1.0, log=log)
    assert ledger.spent == pytest.approx(1.2)
    with pytest.raises(RuntimeError, match="cap"):
        ledger.check()
    # Rebuilt from the log on first use — that is the recovery path.
    assert SpendLedger(cap=1.0, log=log).spent == pytest.approx(1.2)


def test_the_cap_refuses_before_admitting(tmp_path):
    log = tmp_path / "calls.jsonl"
    log.write_text(json.dumps({"usd": 99.0}) + "\n", encoding="utf-8")
    gw = Gateway([Backend(url="http://a/v1", model="m")], spend_cap=1.0, calls_log=log)
    status, payload, _ = gw.forward(b"{}", {})
    assert status == 402 and b"cap" in payload


def test_draining_stops_admitting():
    gw = Gateway([Backend(url="http://a/v1", model="m")])
    gw._draining.set()
    status, _, _ = gw.forward(b"{}", {})
    assert status == 503


# ── replica sizing ───────────────────────────────────────────────────────────


def test_auto_replicas_is_vram_first_shard_count_second():
    """Shards do not create capacity. 4 shards → 1 replica, 8 → 2, but CAPPED by
    what actually fits — and a machine that holds one gets one, with the extra
    shards waiting in the gateway queue, which is correct and no longer fatal."""
    gb = 1024 ** 3
    n, bounds = auto_replicas(vram_free_bytes=48 * gb, model_bytes=5 * gb, shards=4)
    assert n == 1 and bounds["by_demand"] == 1
    n, _ = auto_replicas(vram_free_bytes=48 * gb, model_bytes=5 * gb, shards=8)
    assert n == 2
    # VRAM binds: eight shards, but only one replica fits.
    n, bounds = auto_replicas(vram_free_bytes=6 * gb, model_bytes=5 * gb, shards=8)
    assert n == 1 and bounds["by_memory"] == 0
    n, _ = auto_replicas(vram_free_bytes=400 * gb, model_bytes=1 * gb, shards=64,
                         max_replicas=4)
    assert n == 4


# ── backend ownership ────────────────────────────────────────────────────────


def test_replicas_never_use_the_desktop_default_port():
    """So a user's own server is never the one a run tears down."""
    ports = serve.plan_ports(4)
    assert serve.DESKTOP_DEFAULT_PORT not in ports
    assert ports[0] == serve.FIRST_REPLICA_PORT
    with pytest.raises(ValueError, match="default"):
        serve.start(serve.DESKTOP_DEFAULT_PORT, "m", 4096)


def test_an_adopted_replica_is_unloaded_never_killed(monkeypatch):
    """On a shared machine the process is a few MB and the weights gigabytes."""
    killed: list[int] = []
    monkeypatch.setattr(serve.os, "kill", lambda pid, sig: killed.append(pid))
    monkeypatch.setattr(serve, "unload", lambda rep: True)

    adopted = serve.Replica(port=11435, model="m", context=4096, pid=999,
                            owned=True, adopted=True)
    assert serve.stop(adopted) is False
    foreign = serve.Replica(port=11436, model="m", context=4096, read_only=True)
    assert serve.stop(foreign) is False
    assert killed == []


def test_a_read_only_adoption_is_never_unloaded():
    foreign = serve.Replica(port=11437, model="m", context=4096, read_only=True)
    assert serve.unload(foreign) is False


def test_a_wrong_model_on_our_port_is_refused(monkeypatch, tmp_path):
    """Never silently re-use a different instrument."""
    monkeypatch.setattr(serve, "OWNERSHIP_FILE", tmp_path / "serve.json")
    serve._save_ownership({"replicas": {"11435": {"pid": 1, "model": "qwen:7b"}}})
    monkeypatch.setattr(serve, "port_in_use", lambda port, host="127.0.0.1": True)
    monkeypatch.setattr(serve, "_loaded_model", lambda port, timeout=5.0: "llama:8b")
    with pytest.raises(RuntimeError, match="different instrument"):
        serve.start(11435, "qwen:7b", 32768)


# ── end to end over a real socket ────────────────────────────────────────────


class _Upstream:
    """A one-endpoint OpenAI-compatible server that counts what it was asked."""

    def __init__(self, delay: float = 0.0) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        self.calls = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):  # noqa: ANN002, ANN202
                return

            def do_GET(self):  # noqa: N802
                self.send_response(200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def do_POST(self):  # noqa: N802
                outer.calls += 1
                time.sleep(delay)
                body = json.dumps({"choices": [{"message": {"content": "hi"},
                                                "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 1,
                                             "completion_tokens": 1}}).encode()
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
def test_eight_concurrent_shards_missing_one_key_are_billed_once():
    """One key, N concurrent shards, one upstream call — over a real socket."""
    up = _Upstream(delay=0.25)
    port = _free_port()
    gw = Gateway([Backend(url=f"http://127.0.0.1:{up.port}/v1", model="m",
                          concurrency=2)],
                 queue_timeout=None, service_timeout=10.0)
    t = threading.Thread(target=serve_forever, args=(gw,), kwargs={"port": port},
                         daemon=True)
    t.start()
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1)
            break
        except OSError:
            time.sleep(0.05)

    results: list[int] = []

    def shot() -> None:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b'{"model":"m","messages":[]}',
            headers={"Content-Type": "application/json",
                     "X-Caster-Cache-Key": "the-same-key"},
            method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            results.append(r.status)

    threads = [threading.Thread(target=shot) for _ in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(30)

    gw.drain()
    up.stop()
    assert results == [200] * 8
    assert up.calls == 1, f"one key, {up.calls} upstream calls — coalescing failed"
    assert gw.stats["service_timeout"] == 0
