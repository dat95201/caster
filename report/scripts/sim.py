"""Counterfactual scheduler simulation on the ceg-mem verdict matrix.

verdicts.jsonl ran every patch against its task's WHOLE pool, so the verdict
of any case on any patch is known and an arbitrary order can be replayed
exactly. The caster pilot could not do this: there the pool is censored past
the first failure.

What the engine does, and therefore what this reproduces:
  * the order is built ONCE per round, from the ledger as it stood when the
    round began -- nothing is re-ranked mid-round
  * probing stops at the first failing case
  * only probed cases update n; only the killer updates k

The order is built LAZILY, one position at a time, and construction stops at
the kill. The unprobed tail changes neither cost nor the ledger, so this is
the same experiment at a fraction of the work.
"""
from __future__ import annotations

import math
import random

PRIOR_A, PRIOR_B = 0.5, 2.0   # "shipped": an unseen case is worth K = 0.2
MIN_COST = 1e-6
EXPLORE_P = 0.15
EPS = 1e-9

POLICIES = ("natural", "random", "kill-only", "cost-only", "km", "coverage",
            "km-cov", "exp")


def popcount(x):
    return bin(x).count("1")


def simulate(pool, patches, cost, covbits, U, policy, seed,
             a=1.0, b=1.0, c=1.0, explore=EXPLORE_P):
    """Returns (total cpu seconds, total probes, probes-to-kill list)."""
    n = len(pool)
    rng = random.Random((seed * 1000003) ^ (hash_str(policy) * 31)
                        ^ int(a * 97 + b * 193 + c * 389) * 7919)
    k = [0] * n
    seen = [0] * n
    cpu_acc = [0.0] * n
    costs = [cost[x] for x in pool]
    nU = max(popcount(U), 1)

    total_cpu = 0.0
    total_probes = 0
    probes = []

    for fails in patches:
        fset = set(fails)
        # ---- ledger snapshot, frozen for the whole round
        K = [(k[i] + PRIOR_A) / (seen[i] + PRIOR_A + PRIOR_B) for i in range(n)]
        mseen = [cpu_acc[i] / seen[i] for i in range(n) if seen[i]]
        cold = max(mseen) if mseen else 1.0
        m = [max(cpu_acc[i] / seen[i], MIN_COST) if seen[i] else max(cold, MIN_COST)
             for i in range(n)]

        if policy == "natural":
            base = None
        elif policy == "random":
            base = None
        elif policy == "kill-only":
            base = K
        elif policy == "cost-only":
            base = [1.0 / m[i] for i in range(n)]
        elif policy == "km":
            base = [K[i] / m[i] for i in range(n)]
        elif policy == "coverage":
            base = [1.0] * n
        elif policy == "km-cov":
            base = [K[i] / m[i] for i in range(n)]
        elif policy == "exp":
            base = [math.exp(a * math.log(max(K[i], EPS)) - c * math.log(m[i]))
                    for i in range(n)]
        else:
            raise ValueError(policy)

        needs_cov = policy in ("coverage", "km-cov") or (policy == "exp" and b != 0.0)
        bexp = 1.0 if policy in ("coverage", "km-cov") else b

        remaining = list(range(n))
        if policy == "random":
            rng.shuffle(remaining)
        C = 0
        killer = None
        probed = 0

        while remaining:
            if policy in ("natural", "random"):
                pick = remaining[0]
            elif rng.random() < explore:
                pick = remaining[rng.randrange(len(remaining))]
            else:
                if needs_cov:
                    best, pick = None, None
                    for i in remaining:
                        g = (popcount(covbits[i] & ~C) + EPS) / (nU + EPS)
                        s = base[i] * (g ** bexp)
                        if best is None or s > best + 1e-15:
                            best, pick = s, i
                else:
                    best, pick = None, None
                    for i in remaining:
                        if best is None or base[i] > best + 1e-15:
                            best, pick = base[i], i
            remaining.remove(pick)
            C |= covbits[pick]
            probed += 1
            total_cpu += costs[pick]
            seen[pick] += 1
            cpu_acc[pick] += costs[pick]
            if pick in fset:
                killer = pick
                break

        if killer is not None:
            k[killer] += 1
        total_probes += probed
        probes.append(probed)

    return total_cpu, total_probes, probes


def hash_str(s):
    h = 2166136261
    for ch in s:
        h = ((h ^ ord(ch)) * 16777619) & 0xFFFFFFFF
    return h
