"""The arena: tasks for which BOTH the per-case cost and the per-case
coverage of the reference were measured, intersected with the complete
verdict matrix from verdicts.jsonl."""
import json

def build():
    M = json.load(open("verdicts_matrix.json"))
    COST = json.load(open("cost_all.json"))
    COV = json.load(open("cov_all.json"))
    out = {}
    for t in sorted(set(COST) & set(COV) & set(M)):
        pool = M[t]["pool"]
        cost, cov = COST[t], COV[t]
        keep = [c for c in pool
                if cost.get(c) and not cost[c]["timeout"] and cov.get(c)]
        if len(keep) < 4:
            continue
        kept = set(keep)
        patches = [sorted(keep.index(pool[i]) for i in p if pool[i] in kept)
                   for p in M[t]["patches"]]
        c = {x: max(cost[x]["cpu"], 1e-6) for x in keep}
        g = {x: frozenset(cov[x]) for x in keep}
        out[t] = (keep, patches, c, g)
    return out
