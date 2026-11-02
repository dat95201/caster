"""Fixtures: a real miniature ConDefects tree, and a model that does not exist.

The benchmark fixture is **on disk and really executed**, not mocked. The whole
point of `exec/sandbox.py` is process isolation, CPU accounting and an error
taxonomy, and a mock would test none of it. It is small enough (two faults, ~24
probes each, programs of three lines) that the full suite still runs in seconds.

The model, by contrast, *is* faked: a test that calls a real endpoint is a test
that fails for reasons that have nothing to do with the code.
"""
from __future__ import annotations

import json
import pathlib
import textwrap
from typing import Any

import pytest

# ── the miniature benchmark ──────────────────────────────────────────────────

CORRECT = "n = int(input())\nprint(n * 2)\n"
# Fails for n >= 5, so the pool splits into a real F2P and a real P2P — which
# is what the stratified split needs in order to be testable at all.
FAULTY = "n = int(input())\nprint(n * 2 if n < 5 else n * 3)\n"

# A second fault in a different coding task. It is wrong only on the SHORT
# inputs, so its F2P footprint is small and its P2P half is large — the shape
# the stratified split has to cope with.
CORRECT_B = "s = input().strip()\nprint(s[::-1])\n"
FAULTY_B = "s = input().strip()\nprint(s[::-1] if len(s) > 2 else s)\n"


def _write(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_corpus(root: pathlib.Path) -> pathlib.Path:
    """A ConDefects-shaped checkout with two faults and 24 probes each."""
    # ── fault 1: abc001_a/1001 ──────────────────────────────────────────────
    code = root / "Code" / "abc001_a" / "Python" / "1001"
    _write(code / "faultyVersion.py", FAULTY)
    _write(code / "correctVersion.py", CORRECT)
    _write(code / "faultLocation.txt", "2\n")
    test = root / "Test" / "abc001" / "A"
    # Two statement samples (small n, so the faulty version passes them —
    # which is the realistic and awkward case) plus 22 random probes.
    for i, n in enumerate([1, 2], start=0):
        _write(test / "in" / f"00_sample_{i:02d}.txt", f"{n}\n")
        _write(test / "out" / f"00_sample_{i:02d}.txt", f"{n * 2}\n")
    for i in range(22):
        n = i + 1          # 1..22 — a mix either side of the n=5 boundary
        _write(test / "in" / f"01_random_{i:02d}.txt", f"{n}\n")
        _write(test / "out" / f"01_random_{i:02d}.txt", f"{n * 2}\n")

    # ── fault 2: abc002_b/2002 ──────────────────────────────────────────────
    code_b = root / "Code" / "abc002_b" / "Python" / "2002"
    _write(code_b / "faultyVersion.py", FAULTY_B)
    _write(code_b / "correctVersion.py", CORRECT_B)
    _write(code_b / "faultLocation.txt", "2\n")
    test_b = root / "Test" / "abc002" / "B"
    words = ["ab", "abc", "racecar", "zz", "hello", "x"] * 4
    _write(test_b / "in" / "00_sample_00.txt", "ab\n")
    _write(test_b / "out" / "00_sample_00.txt", "ba\n")
    for i, w in enumerate(words):
        _write(test_b / "in" / f"01_random_{i:02d}.txt", f"{w}\n")
        _write(test_b / "out" / f"01_random_{i:02d}.txt", f"{w[::-1]}\n")

    _write(root / "date.txt", "abc001_a 2023-05-01\nabc002_b 2023-06-01\n")
    _write(root / "difficulty.txt", "abc001_a 100\nabc002_b 220\n")
    return root


@pytest.fixture(scope="session")
def corpus_root(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    return build_corpus(tmp_path_factory.mktemp("condefects"))


@pytest.fixture()
def adapter(corpus_root: pathlib.Path):
    from caster.bench.adapter import ConDefectsAdapter

    return ConDefectsAdapter(root=corpus_root, timeout=10.0)


@pytest.fixture()
def frozen(adapter, tmp_path: pathlib.Path) -> dict[str, Any]:
    """A frozen corpus over the miniature pool. `min_loop_pool` lowered to 6."""
    from caster.bench.gate import GateConfig, freeze, install

    doc = freeze(adapter, cfg=GateConfig(min_loop_pool=6, date_min="2000-01-01",
                                         holdout=0.30, seed=1))
    install(adapter, doc)
    return doc


@pytest.fixture()
def run_dir(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Point every store and every output at a throwaway tree."""
    import importlib

    from caster import paths

    monkeypatch.setenv("CASTER_ROOT", str(tmp_path))
    monkeypatch.setenv("RUN_DIR", "t")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("VERDICT_DIR", str(tmp_path / "verdicts"))
    monkeypatch.setenv("PATCH_DIR", str(tmp_path / "patches"))
    importlib.reload(paths)
    yield paths
    monkeypatch.delenv("CASTER_ROOT", raising=False)
    monkeypatch.delenv("RUN_DIR", raising=False)
    monkeypatch.delenv("CACHE_DIR", raising=False)
    monkeypatch.delenv("VERDICT_DIR", raising=False)
    monkeypatch.delenv("PATCH_DIR", raising=False)
    importlib.reload(paths)


# ── the model that does not exist ────────────────────────────────────────────


class FakeClient:
    """Duck-types `LLMClient.complete`. Records every prompt it was given.

    `replies` maps a *round index* to the source it returns, so a test can make
    round 3 the one that repairs the program. Anything unmapped returns the
    faulty source unchanged, which the oracle refutes — the common case.
    """

    def __init__(self, default_source: str, replies: dict[int, str] | None = None,
                 *, model: str = "fake-1", raises: dict[int, Exception] | None = None,
                 k_candidates: int = 1) -> None:
        self.model = model
        self.base_url = None
        self.context_tokens = 0
        self.default_source = default_source
        self.replies = replies or {}
        self.raises = raises or {}
        self.k_candidates = k_candidates
        self.prompts: list[str] = []
        self.nonces: list[str] = []
        self.calls = 0
        self.live_calls = 0
        self.sessions = 0

    def register_session(self) -> bool:
        self.sessions += 1
        return True

    def unregister_session(self) -> bool:
        self.sessions -= 1
        return True

    def _round_of(self, nonce: str) -> int:
        try:
            return int(nonce.rsplit("|r", 1)[1])
        except (IndexError, ValueError):
            return 0

    def complete(self, prompt: str, *, nonce: str, max_tokens: int = 1024,
                 model: str | None = None, temperature: float | None = None,
                 reasoning_effort: str | None = None, meta: Any = None,
                 from_cache_only: bool = False) -> str:
        self.prompts.append(prompt)
        self.nonces.append(nonce)
        self.calls += 1
        self.live_calls += 1
        r = self._round_of(nonce)
        if r in self.raises:
            raise self.raises[r]
        source = self.replies.get(r, self.default_source)
        if meta is not None:
            meta.cache_key = f"fake-{nonce}"
            meta.model = model or self.model
            meta.finish_reason = "stop"
            meta.prompt_tokens = len(prompt) // 4
            meta.completion_tokens = len(source) // 4
        # A real model returns k programs only when slot S8 asked for them.
        # Returning 5 blocks to a prompt that asked for one would let a test
        # pass while the budget-sizing bug it is checking is still there.
        if self.k_candidates > 1 and "candidate corrections" in prompt:
            blocks = []
            for i in range(self.k_candidates):
                blocks.append(
                    f"### Candidate {i + 1} (probability: {1 / (i + 2):.2f})\n"
                    f"```python\n{source}```"
                )
            return "\n\n".join(blocks)
        return f"```python\n{source}```"


class StubSDK:
    """The smallest thing `LLMClient._sdk()` will accept.

    Used where the REAL client has to be exercised — the on-disk response cache,
    the ledger write, and the gateway stamp that tells the client not to write a
    second ledger row. A FakeClient cannot test any of those, because it is the
    thing they live inside.
    """

    def __init__(self, source: str, *, extra: dict | None = None) -> None:
        self.source = source
        self.extra = extra
        self.calls: list[dict] = []
        outer = self

        class _Completions:
            def create(self, **kwargs):          # noqa: ANN003, ANN201
                outer.calls.append(kwargs)
                return outer._reply()

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()

    def _reply(self):                            # noqa: ANN201
        import types

        usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=13,
                                      completion_tokens_details=None)
        choice = types.SimpleNamespace(
            message=types.SimpleNamespace(content=f"```python\n{self.source}```"),
            finish_reason="stop")
        reply = types.SimpleNamespace(choices=[choice], usage=usage)
        reply.model_extra = self.extra
        return reply


@pytest.fixture()
def fake_client() -> type[FakeClient]:
    return FakeClient


@pytest.fixture()
def stub_sdk() -> type[StubSDK]:
    return StubSDK


@pytest.fixture()
def stores(run_dir):
    from caster.exec.sandbox import protocol_fields
    from caster.exec.store import PatchStore, VerdictStore

    return VerdictStore(protocol_fields()), PatchStore()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: exercises the sandbox heavily")


__all__ = ["build_corpus", "FakeClient", "StubSDK", "CORRECT", "FAULTY", "CORRECT_B", "FAULTY_B",
           "json", "textwrap"]
