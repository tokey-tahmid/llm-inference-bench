"""Print length-sweep summary numbers to stdout for hand-written writeups.

    python analysis/scan_length_sweep.py

Reads results/raw/ and prints per-(backend, TP, concurrency, input, output)
medians of TTFT p50, ITL per token, and output tok/s. Every number traces to
committed raw artifacts. This is a read-only inspection: it writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_results import load  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS_RAW = REPO / "results" / "raw"


def main() -> int:
    frame, report = load(RESULTS_RAW, include_warmup=False, include_failed=False)
    print(report.summary())
    print()

    ok = frame[
        (frame["status"] == "ok") & (frame["sweep_name"] == "length-sweep")
    ]
    if ok.empty:
        print("no length-sweep artifacts")
        return 1

    keys = ["backend", "tensor_parallel_size", "concurrency", "input_len", "output_len"]
    grouped = ok.groupby(keys, dropna=False)
    print(
        f"{'backend':6} {'TP':>2} {'c':>3} {'in':>5} {'out':>5} {'N':>2}  "
        f"{'TTFT_p50(s)':>13} {'ITL_pt(ms)':>11} {'tput(tok/s)':>12}"
    )
    for keys_tuple, grp in sorted(grouped.groups.items()):
        sub = ok.loc[grouped.groups[keys_tuple]]
        b, tp, c, il, ol = keys_tuple
        ttft = sub["ttft_s_p50"].median()
        itl = sub["itl_per_token_s_p50"].dropna().median()
        tput = sub["output_tokens_per_s"].median()
        n = len(sub)

        def f(v: object, d: int = 3) -> str:
            if v is None or (isinstance(v, float) and v != v):
                return "—"
            return f"{v:.{d}g}"

        print(
            f"{b:6} {tp:>2g} {c:>3g} {il:>5g} {ol:>5g} {n:>2}  "
            f"{f(ttft, 4):>13} {f(itl * 1000 if itl == itl else None, 4):>11} "
            f"{f(tput, 4):>12}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
