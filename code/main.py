"""One-click entry: Q1 -> Q2 -> Q3 -> Q4 -> Q5 -> summary.

Usage (from the project root):  python code/main.py [--data-dir PATH] [--only 1 2 ...]
Each step reuses its validated caches (same inputs, code hash and parameters) and otherwise recomputes.
"""
import os, sys, argparse, time, importlib
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="folder containing 'Problem A Dataset 1/2' and '公开数据'")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="run only these problems (1-5)")
    a = ap.parse_args()
    if a.data_dir:
        os.environ["FLOOD_DATA_DIR"] = os.path.abspath(a.data_dir)
    steps = [(0, "fetch_worldcover"), (1, "problem1"), (1, "coreg_official"), (2, "problem2"), (3, "problem3"), (4, "problem4"), (5, "problem5")]
    for n, mod in steps:
        if a.only and n not in a.only and n != 0:
            continue
        t0 = time.time()
        print(f"=== {mod} ===", flush=True)
        try:
            importlib.import_module(mod).main()
        except Exception as e:      # one failing problem must not hide the others' results
            print(f"[main] {mod} FAILED: {type(e).__name__}: {e}", flush=True)
            if n == 0:
                print("[main] WorldCover download failed; Q1 falls back to optical index rules", flush=True)
                continue
            raise
        print(f"=== {mod} finished in {time.time() - t0:.0f}s ===", flush=True)
    importlib.import_module("summarize").main()
    importlib.import_module("constraint_audit").main()


if __name__ == "__main__":
    main()
