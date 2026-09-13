# search_bench.py
"""
A/B benchmark harness for the Searcher.

Answers two questions:
  1) Is a change faster?   → paired t-test on (uninstrumented) search time
  2) WHY is it faster?     → nodes, nodes/sec, TT hit rate, cutoffs/node,
                             plus an optional per-function profile pass

One entry point: compare_search().

    from search_bench import compare_search

    # single config: where does the time go?
    compare_search(depth=5, runs=3)

    # A/B a change — subclass, plain class, or factory(board) -> Searcher
    class NoLMR(Searcher):
        def _get_reduction(self, move, number, depth): return 0

    res = compare_search(candidate=NoLMR, depth=5, runs=5)
    res["summary"]["significant"]   # bool
    res["summary"]["pct"]           # % time change, negative = faster

Methodology (what makes the numbers trustworthy):
  * Paired: both sides search identical positions; who runs first
    alternates (ABBA-style), so machine drift over time can't favour
    one side.
  * Warmup runs are discarded (CPU governor, allocator, caches).
  * gc is disabled while timing.
  * Fixed depth by default → node counts are deterministic and directly
    comparable. Pass time_limit=... only to test time-management changes
    (both sides then spend the whole budget; compare nodes/last_depth,
    not wall time).
  * The headline timing is UN-instrumented. The optional profile pass
    re-runs with per-function timers (wrapper overhead included, both
    sides equally) — never use profile-pass times as the verdict.
  * Statistics via scipy (stats.t for p-values and critical values);
    t itself is computed manually so zero-variance differences (common
    with deterministic node counts) don't produce NaNs.

Caveats:
  * Run on an idle machine; background load is the main noise source.
  * Each sample uses a fresh Searcher, so this measures single-search
    speed, not cross-move TT reuse in a real game.
  * The evaluation object is shared across all samples (set_board
    rebinding handles the normal case).
  * Profile-pass µs/call figures include ~0.5µs of wrapper overhead.
  * Assumes Searcher.search(depth, time_limit=None) from the
    time-management step.
"""
from __future__ import annotations

import gc
import math
from collections import defaultdict
from statistics import median, stdev
from time import perf_counter

import bulletchess as bc
from scipy import stats

from evaluation import Evaluation, Handcrafted, NNEvaluation

from searcher import Searcher   # ← adjust to your module name

__all__ = ["compare_search"]

# (name, FEN) — opening, sharp middlegames, promotions, endgame
DEFAULT_POSITIONS = [
    ("startpos",  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("kiwipete",  "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
    ("midgame",   "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("promos",    "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1"),
    ("endgame",   "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ("cpw5",      "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8"),
]

# Timed in the profile pass. Deliberately mutually NON-nested (none of
# these call each other) so their times add up without double counting.
# _move_tactical_score / _get_reduction are NOT here because they call
# `see` / `_gives_check` — their remaining cost lands in "(untimed)".
PROFILE_FUNCS = ("_tt_lookup", "_cache", "see", "_gives_check")


# ─────────────────────────── statistics ────────────────────────────

def _paired_t(diffs, alpha=0.05):
    """Two-sided paired t-test on a list of (B - A) differences.

    Returns mean, sd, t, two-tailed p, df, and crit (two-tailed
    critical value at alpha). scipy provides p and crit; t itself is
    computed manually because we need mean/sd anyway for the CI, and
    because zero-variance diffs (identical node counts at fixed depth)
    need graceful handling instead of nan.
    """
    n = len(diffs)
    mean = sum(diffs) / n
    if n < 2:
        return {"t": None, "p": None, "df": 0, "mean": mean,
                "sd": 0.0, "crit": None}
    df = n - 1
    sd = stdev(diffs)                        # sample stdev (ddof=1)
    if sd == 0.0:
        # Constantly hit with deterministic node counts: all diffs are 0
        t = 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
        p = 1.0 if mean == 0.0 else 0.0
    else:
        t = mean / (sd / math.sqrt(n))
        p = 2.0 * stats.t.sf(abs(t), df)     # two-tailed p-value
    crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    return {"t": t, "p": p, "df": df, "mean": mean, "sd": sd, "crit": crit}


# ───────────────────────── small helpers ───────────────────────────

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _pct(new, old):
    return (new - old) / old * 100.0 if old else 0.0


def _fmt_dur(x):
    if x >= 10:
        return f"{x:,.2f} s"
    if x >= 0.01:
        return f"{x * 1000:,.0f} ms"
    return f"{x * 1e6:,.0f} µs"


def _fmt_p(p):
    if p is None:
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def _fmt_t(t):
    if t is None:
        return "n/a"
    if math.isinf(t):
        return "∞"
    return f"{t:.2f}"


# ─────────────────────── input normalization ───────────────────────

def _normalize_positions(positions):
    out = []
    for i, p in enumerate(positions):
        if isinstance(p, tuple) and len(p) == 2 and isinstance(p[1], str):
            name, fen = p
        elif isinstance(p, str):
            name, fen = f"pos{i + 1}", p
        elif isinstance(p, bc.Board):
            name, fen = f"pos{i + 1}", p.fen()
        else:
            raise TypeError(f"position {i}: expected FEN, (name, FEN), or bc.Board")
        try:
            bc.Board.from_fen(fen)
        except Exception as exc:
            raise ValueError(f"position {i} ({name}): invalid FEN {fen!r}") from exc
        out.append((name, fen))
    if not out:
        raise ValueError("no positions to benchmark")
    return out


def _resolve_evaluation(spec):
    if spec is None:
        return Handcrafted()
    if isinstance(spec, Evaluation):
        return spec                      # shared instance (set_board rebinds)
    if isinstance(spec, type) or callable(spec):
        return spec()                    # class or zero-arg factory
    raise TypeError("evaluation must be an Evaluation instance, subclass, "
                    "factory, or None")


def _make_factory(spec, evaluation, default=False):
    """Normalize a searcher spec into a factory(board) -> Searcher."""
    if spec is None:
        if not default:
            return None
        return lambda board: Searcher(board, evaluation)
    if isinstance(spec, type):           # Searcher (sub)class
        return lambda board: spec(board, evaluation)
    if callable(spec):                   # factory taking a board
        return spec
    raise TypeError("baseline/candidate must be None, a Searcher subclass, "
                    "or a callable(board) -> Searcher")


# ─────────────── instrumentation (zero cost when unused) ───────────
#
# We monkey-patch timed wrappers onto a FRESH searcher instance right
# before the profile pass and throw the instance away afterwards.
# The normal timing pass never touches these, and the Searcher source
# has no profiling code in its hot loops at all.

def _wrap_timed(obj, name, timers):
    original = getattr(obj, name)
    slot = timers.setdefault(name, [0.0, 0])   # [seconds, calls]

    def timed(*args, __original=original, __slot=slot, **kwargs):
        t0 = perf_counter()
        try:
            return __original(*args, **kwargs)
        finally:
            __slot[0] += perf_counter() - t0
            __slot[1] += 1

    setattr(obj, name, timed)


class _TimedEval:
    """Times Evaluation.__call__/do/undo; forwards everything else."""

    __slots__ = ("_inner", "_timers")

    def __init__(self, inner, timers):
        self._inner = inner
        self._timers = timers

    def __call__(self, *args, **kwargs):
        s = self._timers.setdefault("evaluate()", [0.0, 0])
        t0 = perf_counter()
        try:
            return self._inner(*args, **kwargs)
        finally:
            s[0] += perf_counter() - t0
            s[1] += 1

    def do(self, move):
        s = self._timers.setdefault("evaluate.do", [0.0, 0])
        t0 = perf_counter()
        try:
            return self._inner.do(move)
        finally:
            s[0] += perf_counter() - t0
            s[1] += 1

    def undo(self):
        s = self._timers.setdefault("evaluate.undo", [0.0, 0])
        t0 = perf_counter()
        try:
            return self._inner.undo()
        finally:
            s[0] += perf_counter() - t0
            s[1] += 1

    def __getattr__(self, name):
        return getattr(self._inner, name)


def _instrument(searcher, timers):
    for name in PROFILE_FUNCS:
        if hasattr(searcher, name):
            _wrap_timed(searcher, name, timers)
    searcher.evaluate = _TimedEval(searcher.evaluate, timers)


# ─────────────────────────── aggregation ───────────────────────────

def _stats_of(searcher):
    if hasattr(searcher, "stats"):
        return searcher.stats()
    lookups = searcher.tt_lookups
    return {
        "nodes": searcher.nodes,
        "tt_lookups": lookups,
        "tt_hits": searcher.tt_hits,
        "tt_hit_rate": searcher.tt_hits / lookups if lookups else 0.0,
        "beta_cutoffs": searcher.beta_cutof,
        "aspiration_fails": searcher.aspr_fail,
        "last_depth": getattr(searcher, "last_depth", 0),
    }


def _aggregate(samples):
    times = [s["time"] for s in samples]
    nodes = [s["nodes"] for s in samples]
    total_time = sum(times)
    total_nodes = sum(nodes)
    lookups = sum(s["tt_lookups"] for s in samples)
    hits = sum(s["tt_hits"] for s in samples)
    cutoffs = sum(s["beta_cutoffs"] for s in samples)
    return {
        "n": len(samples),
        "time_mean": _mean(times),
        "time_median": median(times),
        "time_stdev": stdev(times) if len(times) > 1 else 0.0,
        "total_time": total_time,
        "nodes_mean": _mean(nodes),
        "nodes_stdev": stdev(nodes) if len(nodes) > 1 else 0.0,
        "total_nodes": total_nodes,
        "nps": total_nodes / total_time if total_time else 0.0,
        "tt_hit_rate": hits / lookups if lookups else 0.0,
        "cutoffs_per_node": cutoffs / total_nodes if total_nodes else 0.0,
        "aspiration_fails": sum(s["aspiration_fails"] for s in samples),
        "last_depth_mean": _mean([s["last_depth"] for s in samples]),
    }


def _diagnose(agg_a, agg_b, stats_nodes, alpha, significant, diff_pct):
    """Human-readable 'why' lines for the time change."""
    lines = []
    node_pct = _pct(agg_b["nodes_mean"], agg_a["nodes_mean"])
    nps_pct = _pct(agg_b["nps"], agg_a["nps"])
    nodes_sig = (stats_nodes is not None and stats_nodes["p"] is not None
                 and stats_nodes["p"] < alpha)
    if abs(node_pct) >= 0.5:
        what = ("better pruning / move ordering / TT use" if node_pct < 0
                else "weaker pruning or worse move ordering")
        lines.append(f"tree size: {node_pct:+.1f}% nodes → {what}"
                     + (" (significant)" if nodes_sig else ""))
    if abs(nps_pct) >= 2.0:
        lines.append(f"raw speed: {nps_pct:+.1f}% nodes/sec → per-node code "
                     f"got {'faster' if nps_pct > 0 else 'slower'}")
    tt_pp = (agg_b["tt_hit_rate"] - agg_a["tt_hit_rate"]) * 100.0
    if abs(tt_pp) >= 0.5:
        lines.append(f"TT hit rate: {tt_pp:+.1f} pp → caching "
                     f"{'improved' if tt_pp > 0 else 'degraded'}")
    co = agg_b["cutoffs_per_node"] - agg_a["cutoffs_per_node"]
    if abs(co) >= 0.002:
        lines.append(f"cutoffs per node: {co:+.3f} → move ordering "
                     f"{'improved' if co > 0 else 'weakened'}")
    if significant and diff_pct is not None:
        if node_pct < -1.0 and abs(nps_pct) < 5.0:
            lines.append("⇒ the time change comes from searching fewer nodes, "
                         "not faster code")
        elif nps_pct > 3.0 and abs(node_pct) < 2.0:
            lines.append("⇒ the time change comes from faster per-node code")
        elif node_pct < -1.0 and nps_pct > 3.0:
            lines.append("⇒ both a smaller tree and faster per-node code")
        elif node_pct > 1.0 and nps_pct < -3.0:
            lines.append("⇒ both a bigger tree and slower per-node code")
        elif node_pct > 1.0:
            lines.append("⇒ the tree got bigger — weaker pruning is the main cost")
    return lines


# ──────────────────────────── main entry ───────────────────────────

def compare_search(baseline=None, candidate=None, evaluation=None,
                   positions=None, depth=6, runs=5, warmup=1,
                   alpha=0.05, profile=True, profile_runs=1,
                   time_limit=None, label_a="baseline",
                   label_b="candidate", verbose=True):
    """
    Benchmark search speed and diagnose WHY it changed.

    baseline / candidate:
        None            -> default Searcher(evaluation)
        Searcher class  -> constructed as cls(board, evaluation)
        factory         -> callable(board) -> Searcher, for config tweaks
                           that aren't expressible as a subclass
        candidate=None  -> single-config run: stats + profile only.

    evaluation:  Evaluation instance / subclass / zero-arg factory / None
                 (None -> Handcrafted()).
    positions:   list of FEN strings, (name, FEN) tuples, or bc.Board.
    depth:       fixed search depth (deterministic node counts).
    runs:        timed repetitions per position (more = tighter statistics).
    warmup:      discarded runs before timing starts.
    alpha:       significance level for the paired t-tests (the printed
                 confidence interval matches it).
    profile:     run an extra instrumented pass for per-function timings.
    profile_runs: repetitions of the profile pass.
    time_limit:  only for testing time management — both sides then spend
                 the budget, so compare nodes / last_depth, not wall time.
    verbose:     print the report (the result dict is always returned).
    """
    fens = _normalize_positions(DEFAULT_POSITIONS if positions is None
                                else positions)
    if runs < 1:
        raise ValueError("runs must be >= 1")
    evaluation = _resolve_evaluation(evaluation)
    fac_a = _make_factory(baseline, evaluation, default=True)
    fac_b = None if candidate is None else _make_factory(candidate, evaluation)
    single = fac_b is None
    if not single and len(fens) * runs < 2:
        raise ValueError("need >= 2 paired samples: positions * runs >= 2")

    # ── pass 1: clean timing (this is the only pass that decides speed) ──
    samples = {"A": [], "B": []}
    gc_was_on = gc.isenabled()
    gc.disable()
    bench_start = perf_counter()
    try:
        for run in range(warmup + runs):
            record = run >= warmup
            for pi, (pname, fen) in enumerate(fens):
                # counterbalance: who goes first alternates, so machine
                # slowdown over the bench can't favour one side
                order = ("A", "B") if (run + pi) % 2 == 0 else ("B", "A")
                for tag in order:
                    if tag == "B" and single:
                        continue
                    factory = fac_a if tag == "A" else fac_b
                    searcher = factory(bc.Board.from_fen(fen))
                    t0 = perf_counter()
                    searcher.search(depth, time_limit=time_limit)
                    elapsed = perf_counter() - t0
                    if record:
                        samples[tag].append(
                            {"position": pname, "run": run - warmup,
                             "time": elapsed, **_stats_of(searcher)})
            if verbose and record:
                print(f"  [run {run - warmup + 1}/{runs} done, "
                      f"{perf_counter() - bench_start:.1f}s elapsed]")
    finally:
        if gc_was_on:
            gc.enable()
    bench_time = perf_counter() - bench_start

    agg_a = _aggregate(samples["A"])
    agg_b = None if single else _aggregate(samples["B"])

    # ── paired statistics ──
    stats_time = stats_nodes = None
    ci_pct = wins = pairs = None
    diff_pct = None
    significant = False
    verdict = None
    if not single:
        by = {tag: {(s["position"], s["run"]): s for s in samples[tag]}
              for tag in ("A", "B")}
        keys = sorted(set(by["A"]) & set(by["B"]))
        pairs = [(by["A"][k], by["B"][k]) for k in keys]
        stats_time = _paired_t([b["time"] - a["time"] for a, b in pairs], alpha)
        stats_nodes = _paired_t([b["nodes"] - a["nodes"] for a, b in pairs], alpha)
        wins = sum(1 for a, b in pairs if b["time"] < a["time"])
        if agg_a["time_mean"] > 0:
            diff_pct = _pct(agg_b["time_mean"], agg_a["time_mean"])
            if stats_time["sd"] > 0.0:
                h = stats_time["crit"] * stats_time["sd"] / math.sqrt(len(pairs))
                ci_pct = ((stats_time["mean"] - h) / agg_a["time_mean"] * 100,
                          (stats_time["mean"] + h) / agg_a["time_mean"] * 100)
            elif stats_time["mean"] != 0.0:      # zero variance, nonzero diff
                ci_pct = (-math.inf, math.inf)
            else:                                # every diff exactly zero
                ci_pct = (0.0, 0.0)
        p = stats_time["p"]
        if p is not None and p < alpha and stats_time["mean"] != 0:
            significant = True
            verdict = "FASTER" if stats_time["mean"] < 0 else "SLOWER"
        else:
            verdict = "no significant difference"

    # ── pass 2: instrumented profile (diagnosis only) ──
    profiles = None
    if profile:
        profiles = {}
        for tag, factory in (("A", fac_a), ("B", fac_b)):
            if factory is None:
                continue
            timers = defaultdict(lambda: [0.0, 0])
            total = 0.0
            for _ in range(profile_runs):
                for pname, fen in fens:
                    searcher = factory(bc.Board.from_fen(fen))
                    local = {}
                    _instrument(searcher, local)
                    t0 = perf_counter()
                    searcher.search(depth)          # never time-limited here
                    total += perf_counter() - t0
                    for k, (sec, calls) in local.items():
                        timers[k][0] += sec
                        timers[k][1] += calls
            profiles[tag] = {"timers": dict(timers), "total": total}

    # ── report ──
    if verbose:
        bar = "─" * 72
        print()
        print(bar)
        print(f" SEARCH BENCHMARK   depth={depth}  runs={runs}  warmup={warmup}  "
              f"positions={len(fens)}  eval={type(evaluation).__name__}")
        if time_limit is not None:
            print(" ⚠ time_limit set: both sides spend ~the full budget —")
            print("   compare nodes / last_depth rather than wall time")
        print(bar)
        print(f" A: {label_a}")
        if not single:
            print(f" B: {label_b}")
        print()

        def row(label, sa, sb="", sd=""):
            print(f" {label:<20}{sa:>16}{sb:>16}{sd:>12}")

        row("metric", label_a[:16], label_b[:16] if not single else "", "diff")
        row("time (mean)", _fmt_dur(agg_a["time_mean"]),
            _fmt_dur(agg_b["time_mean"]) if not single else "",
            f"{diff_pct:+.1f}%" if diff_pct is not None else "")
        row("time (median)", _fmt_dur(agg_a["time_median"]),
            _fmt_dur(agg_b["time_median"]) if not single else "")
        row("time (stdev)", _fmt_dur(agg_a["time_stdev"]),
            _fmt_dur(agg_b["time_stdev"]) if not single else "")
        row("nodes (mean)", f"{agg_a['nodes_mean']:,.0f}",
            f"{agg_b['nodes_mean']:,.0f}" if not single else "",
            f"{_pct(agg_b['nodes_mean'], agg_a['nodes_mean']):+.1f}%" if not single else "")
        row("nodes / sec", f"{agg_a['nps']:,.0f}",
            f"{agg_b['nps']:,.0f}" if not single else "",
            f"{_pct(agg_b['nps'], agg_a['nps']):+.1f}%" if not single else "")
        row("TT hit rate", f"{agg_a['tt_hit_rate'] * 100:.1f}%",
            f"{agg_b['tt_hit_rate'] * 100:.1f}%" if not single else "",
            f"{(agg_b['tt_hit_rate'] - agg_a['tt_hit_rate']) * 100:+.1f}pp" if not single else "")
        row("cutoffs / node", f"{agg_a['cutoffs_per_node']:.3f}",
            f"{agg_b['cutoffs_per_node']:.3f}" if not single else "",
            f"{agg_b['cutoffs_per_node'] - agg_a['cutoffs_per_node']:+.3f}" if not single else "")
        row("last depth (mean)", f"{agg_a['last_depth_mean']:.1f}",
            f"{agg_b['last_depth_mean']:.1f}" if not single else "")

        per_pos = defaultdict(lambda: ([], []))
        for s in samples["A"]:
            per_pos[s["position"]][0].append(s["nodes"])
        for s in samples["B"]:
            per_pos[s["position"]][1].append(s["nodes"])
        det = all(stdev(xs) == 0.0
                for lists in per_pos.values() for xs in lists if len(xs) > 1)
        print(f"\n node counts "
              f"{'deterministic across runs ✓' if det else 'VARY between runs (nondeterministic search?)'}")

        if not single:
            print(f"\n paired t-test on time (n={len(pairs)}): "
                  f"t={_fmt_t(stats_time['t'])}, df={stats_time['df']}, "
                  f"p={_fmt_p(stats_time['p'])}")
            if ci_pct is not None:
                print(f" {(1 - alpha) * 100:.0f}% CI of mean time diff: "
                      f"[{ci_pct[0]:+.1f}%, {ci_pct[1]:+.1f}%] of A's mean")
            print(f" B faster than A on {wins}/{len(pairs)} position-runs")
            if significant:
                print(f"\n → {label_b} is {verdict} — statistically "
                      f"significant at α={alpha}")
            else:
                print(f"\n → {verdict} at α={alpha} "
                      "(effect ~0, or you need more runs)")
            diag = _diagnose(agg_a, agg_b, stats_nodes, alpha,
                             significant, diff_pct)
            if diag:
                print("\n what changed:")
                for d in diag:
                    print(f"  • {d}")

        print("\n per-position (mean over runs):")
        per = defaultdict(lambda: ([], []))
        for s in samples["A"]:
            per[s["position"]][0].append(s["time"])
        for s in samples["B"]:
            per[s["position"]][1].append(s["time"])
        print(f" {'position':<22}{'A':>14}{'B':>14}{'diff':>10}")
        for name, _ in fens:
            ta, tbs = per[name]
            d = f"{_pct(_mean(tbs), _mean(ta)):+.1f}%" if (tbs and ta) else ""
            print(f" {name:<22}{_fmt_dur(_mean(ta)):>14}"
                  f"{(_fmt_dur(_mean(tbs)) if tbs else ''):>14}{d:>10}")

        if profiles:
            pa, pb = profiles.get("A"), profiles.get("B")
            print("\n profile pass (instrumented — wrapper overhead included;"
                  " diagnosis only, never the verdict):")
            hdr = f" {'function':<16}{'A total':>10}{'A µs/c':>8}{'A %':>6}"
            if pb:
                hdr += f"{'B total':>10}{'B µs/c':>8}{'B %':>6}"
            print(hdr)

            def cells(sec, calls, total):
                us = sec / calls * 1e6 if calls else 0.0
                share = sec / total * 100 if total else 0.0
                return f"{_fmt_dur(sec):>10}{us:>8.1f}{share:>6.1f}"

            names = sorted(set(pa["timers"])
                           | (set(pb["timers"]) if pb else set()))
            for name in names:
                asec, acal = pa["timers"].get(name, (0.0, 0))
                line = f" {name:<16}{cells(asec, acal, pa['total'])}"
                if pb:
                    bsec, bn = pb["timers"].get(name, (0.0, 0))
                    line += cells(bsec, bn, pb["total"])
                print(line)
            other_a = pa["total"] - sum(v[0] for v in pa["timers"].values())
            line = f" {'(untimed)':<16}{cells(other_a, 0, pa['total'])}"
            if pb:
                other_b = pb["total"] - sum(v[0] for v in pb["timers"].values())
                line += cells(other_b, 0, pb["total"])
            print(line + "   ← apply/undo, movegen, hash, heap, interpreter")
            print(f" {'TOTAL':<16}{cells(pa['total'], 0, pa['total'])}"
                  + (cells(pb["total"], 0, pb["total"]) if pb else ""))

        print(f"\n total bench wall time: {_fmt_dur(bench_time)}")
        print(bar)

    return {
        "config": {"depth": depth, "runs": runs, "warmup": warmup,
                   "positions": [n for n, _ in fens], "alpha": alpha,
                   "evaluation": type(evaluation).__name__,
                   "time_limit": time_limit},
        "labels": {"A": label_a, "B": None if single else label_b},
        "summary": {"time_a": agg_a["time_mean"],
                    "time_b": None if single else agg_b["time_mean"],
                    "pct": diff_pct,
                    "p": None if single else stats_time["p"],
                    "t": None if single else stats_time["t"],
                    "df": None if single else stats_time["df"],
                    "ci_pct": ci_pct, "significant": significant,
                    "verdict": verdict},
        "nodes": {"a": agg_a["nodes_mean"],
                  "b": None if single else agg_b["nodes_mean"],
                  "p": None if single else stats_nodes["p"],
                  "deterministic": agg_a["nodes_stdev"] == 0.0
                  and (single or agg_b["nodes_stdev"] == 0.0)},
        "nps": {"a": agg_a["nps"], "b": None if single else agg_b["nps"]},
        "tt_hit_rate": {"a": agg_a["tt_hit_rate"],
                        "b": None if single else agg_b["tt_hit_rate"]},
        "cutoffs_per_node": {"a": agg_a["cutoffs_per_node"],
                             "b": None if single else agg_b["cutoffs_per_node"]},
        "samples": {"A": samples["A"], "B": samples["B"]},
        "profile": profiles,
        "bench_time": bench_time,
    }


if __name__ == "__main__":
    # Example 1 — single config: where does the time go?
    # compare_search(depth=5, runs=3, label_a="current searcher")

    # Example 2 — A/B: does late-move reduction actually pay off here?
    class FastSEE:
        """SEE via attack queries + apply/undo, with optional recaptures.
    
        Pseudo-legal by design (pinned 'attackers' are counted — standard
        for SEE, same trade-off every C engine makes).
        """
        PIECE_VALUES = {
            bc.PAWN: 100, bc.ROOK: 500, bc.KNIGHT: 320,
            bc.BISHOP: 330, bc.QUEEN: 900, bc.KING: 2000,
        }
    
        def __init__(self, board: bc.Board, piece_values: dict = None):
            if not hasattr(bc.Board, "attacks_to"):     # verify this API exists in your version!
                raise NotImplementedError("bulletchess has no attacks_to — check the attack-query API name")
            self.board = board
            self.pieces_values = piece_values or self.PIECE_VALUES
    
        def see(self, move: bc.Move) -> int:
            board = self.board
            if not move.is_capture(board):
                return 0
    
            to = move.destination
            captured = board[to]
            if captured is None:                          # en passant
                cap_sq = to.south() if board.turn == bc.WHITE else to.north()
                captured = board[cap_sq]
            victim = self.pieces_values[captured.piece_type]
    
            board.apply(move)
            try:
                return victim - self._exchange_gain(to)
            finally:
                board.undo()
    
        def _exchange_gain(self, to) -> int:
            """Best net gain for the side to move from continuing the exchange
            on `to`. 0 = declining (or having no attacker) is at least as good."""
            board = self.board
            side = board.turn
            best = 0
            for sq in board.attacks_to(to, side):
                piece = board[sq]
                if piece is None:
                    continue
                if (piece.piece_type == bc.KING
                        and board.attacks_to(to, side.opposite)):
                    continue                              # king can't recapture into fire
                mv = bc.Move(sq, to)
                if (piece.piece_type == bc.PAWN
                        and to.index() // 8 == (7 if piece.color == bc.WHITE else 0)):
                    mv = bc.Move(sq, to, bc.QUEEN)        # promotions valued correctly
                board.apply(mv)
                try:
                    gain = (self.pieces_values[board[to].piece_type]
                            - self._exchange_gain(to))
                finally:
                    board.undo()
                if gain > best:
                    best = gain
            return best

    import heapq
    class UesefastSEE(Searcher):
        def quiesce(self, alpha: float, beta: float, ply: int) -> float:
            if self._out_of_time():  # --- time management ---
                return 0
            self.nodes += 1
            if self.board in bc.MATE:
                return self._terminal_score(ply)
    
            key = hash(self.board)
            cached = self._tt_lookup(key, 0, alpha, beta, False)
    
            if cached is not None:
                return cached
    
            orig_alpha = alpha
            pv = self.pieces_values
    
            if self.board in bc.CHECK:
                best_value = float("-inf")
                moves = self.board.legal_moves()
            else:
                    
                    
                stand_pat = self.evaluate()
                
                if stand_pat >= beta:
                    return stand_pat
    
                alpha = max(alpha, stand_pat)
                best_value = stand_pat
    
                if stand_pat + self.evaluate.queen < alpha:
                    self._cache(key, 0, best_value, None, orig_alpha, beta, False)
                    return best_value
    
                moves = []
                for move in self.board.legal_moves():
                    if not (move.is_capture(self.board) or move.promotion):
                        continue
    
                    victim = self.board[move.destination]
                    gain = pv[victim.piece_type] if victim else 0
                    if move.promotion:
                        gain += pv[move.promotion]
    
                    if stand_pat + gain + 200 < alpha:
                        continue
                    if self.see(move) < 0:
                        continue
                    moves.append(move)
    
            entry = self.tt.get(key)
            best = entry[1] if entry else None
            moves = [
                (-self._move_tactical_score(m, best, ply), i, m)
                for i, m in enumerate(moves)
            ]
            heapq.heapify(moves)
    
            best_move = None
            first_move = True
            while moves:
                _, _, move = moves[0]
    
                self.evaluate.do(move)
                self.board.apply(move)
                if first_move:
                    value = -self.quiesce(-beta, -alpha, ply + 1)
                    first_move = False
                else:
                    value = -self.quiesce(-alpha - 1, -alpha, ply + 1)
                    # --- time management: don't start a re-search if we're aborting ---
                    if alpha < value < beta and not self.stop:
                        value = -self.quiesce(-beta, -alpha, ply + 1)
                self.board.undo()
                self.evaluate.undo()
    
                if self.stop:  # --- time management: unwind, discard value ---
                    break
    
                if value > best_value:
                    best_value = value
                    best_move = move
                    alpha = max(alpha, best_value)
    
                if alpha >= beta:
                    self.beta_cutof += 1
                    break
                heapq.heappop(moves)
            self._cache(key, 0, best_value, best_move, orig_alpha, beta, False)
    
            return best_value
            
    result = compare_search(candidate=UesefastSEE, depth=5, runs=7,
                            label_a="see ordering", label_b="MVVLVA in ordering", evaluation=NNEvaluation)
    print("significant:", result["summary"]["significant"])
    print("time change:", round(result["summary"]["pct"] or 0.0, 1), "%")