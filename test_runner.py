#!/usr/bin/env python3
"""
Practice-case test runner for the btc-cent-forecast entry.

Runs the real program exactly the way the grader does -- one .in file per
invocation, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR pointing at throwaway
directories -- then reports

  * the forecast produced for every case,
  * the persistence baseline (last close) for every case,
  * the output-contract check (finite, > 0, < 1e7, rounded to 5 decimals),
  * and, when the practice cases are mutually overlapping sliding windows
    (they are: 6 x 72 h windows shifted by 4 h), the realised error of both
    models against the ground truth recoverable from the *next* window.

Usage:
    python3 test_runner.py                       # dirs under /tmp (per spec)
    MAKEFUN_TEST_ROOT=~/scratch python3 test_runner.py
"""

import datetime
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN = os.path.join(HERE, "main.py")


def test_root():
    return os.path.abspath(os.path.expanduser(os.environ.get("MAKEFUN_TEST_ROOT", "/tmp")))


def load_cases(repo):
    cases = []
    for name in sorted(os.listdir(repo)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(repo, name), "r", encoding="utf-8") as fh:
            cases.append((name, json.load(fh)))
    return cases


def parse_ts(s):
    return datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def build_timeline(cases, repo):
    """Merge every case's minute series into one timestamp -> price map."""
    tl = {}
    for name, data in cases:
        for m in data.get("markets", []):
            if not m.get("start"):
                continue
            t0 = parse_ts(m["start"])
            for i, p in enumerate(m["prices"]):
                tl[t0 + datetime.timedelta(minutes=i)] = float(p)
    return tl


def ground_truth(cases, repo, timeline):
    """A case's realised target may be visible in another (later/overlapping) window."""
    gt = {}
    for name, data in cases:
        horizon = int(data.get("horizon_minutes", 240) or 240)
        for m in data.get("markets", []):
            if not m.get("start"):
                continue
            t0 = parse_ts(m["start"])
            last_ts = t0 + datetime.timedelta(minutes=len(m["prices"]) - 1)
            target_ts = last_ts + datetime.timedelta(minutes=horizon)
            if target_ts in timeline:
                gt[name[:-3] if name.endswith(".in") else name] = (
                    float(m["prices"][-1]), timeline[target_ts], horizon)
    return gt


def check_contract(path):
    """Return (ok, price, problems)."""
    problems = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        doc = json.loads(raw)
    except Exception as exc:
        return False, None, ["unreadable/invalid JSON: %s" % exc]
    if not isinstance(doc, dict) or not isinstance(doc.get("forecasts"), list):
        return False, None, ["missing 'forecasts' list"]
    if len(doc["forecasts"]) != 1:
        problems.append("expected exactly 1 forecast, got %d" % len(doc["forecasts"]))
    f = doc["forecasts"][0] if doc["forecasts"] else {}
    if not isinstance(f.get("case"), str) or not f.get("case"):
        problems.append("bad/missing 'case'")
    price = f.get("price")
    if not isinstance(price, (int, float)) or isinstance(price, bool):
        problems.append("price is not a number")
        return False, None, problems
    if abs(price) != abs(round(price, 5)):
        problems.append("price not rounded to 5 decimals")
    if not (price > 0):
        problems.append("price not > 0")
    if not (price < 10_000_000):
        problems.append("price not < 10000000")
    return not problems, price, problems


def main():
    repo = HERE
    cases = load_cases(repo)
    if not cases:
        print("no .in files found in", repo)
        return 2

    root = test_root()
    in_dir = os.path.join(root, "input")
    out_dir = os.path.join(root, "output")
    for d in (in_dir, out_dir):
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d, exist_ok=True)

    timeline = build_timeline(cases, repo)
    gt = ground_truth(cases, repo, timeline)

    print("input dir :", in_dir)
    print("output dir:", out_dir)
    print("python    :", sys.version.split()[0])
    print()
    hdr = "%-6s %-11s %-11s %-11s %-9s %s" % (
        "case", "last(persist)", "model", "actual", "|persist|", "|model|")
    print(hdr)
    print("-" * len(hdr))

    ok_all = True
    rows = []
    for name, data in cases:
        stem = name[:-3]
        # exactly one .in file per run, like the grader
        for f in os.listdir(in_dir):
            os.remove(os.path.join(in_dir, f))
        shutil.copy(os.path.join(repo, name), os.path.join(in_dir, name))

        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = in_dir
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        proc = subprocess.run([sys.executable, MAIN], env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            print("%-6s PROGRAM FAILED rc=%d\n%s\n%s" % (stem, proc.returncode,
                                                         proc.stdout, proc.stderr))
            ok_all = False
            continue
        if proc.stderr.strip():
            print("  [stderr] %s" % proc.stderr.strip())

        out_path = os.path.join(out_dir, stem + ".out")
        if not os.path.exists(out_path):
            print("%-6s NO OUTPUT FILE at %s" % (stem, out_path))
            ok_all = False
            continue
        contract_ok, price, problems = check_contract(out_path)
        if not contract_ok:
            ok_all = False
            print("%-6s CONTRACT ERROR: %s" % (stem, "; ".join(problems)))

        prices = data["markets"][0]["prices"]
        last = float(prices[-1])
        if stem in gt:
            _, actual, _ = gt[stem]
            pe, me = abs(last - actual), abs(price - actual)
            rows.append((pe, me))
            mark = "WIN " if me < pe - 1e-9 else ("tie " if abs(me - pe) < 1e-9 else "loss")
            print("%-6s %-11.2f %-11.2f %-11.2f %-9.2f %.2f  %s" % (
                stem, last, price, actual, pe, me, mark))
        else:
            print("%-6s %-11.2f %-11.2f %-11s %-9s %s" % (
                stem, last, price, "(n/a)", "(n/a)", "(no visible ground truth)"))

    print()
    print("contract checks:", "ALL PASS" if ok_all else "FAILURES ABOVE")
    if rows:
        pm = sum(r[0] for r in rows) / len(rows)
        mm = sum(r[1] for r in rows) / len(rows)
        wins = sum(1 for pe, me in rows if me < pe - 1e-9)
        print("evaluated cases with ground truth:", len(rows))
        print("mean |err|  persistence : %.2f" % pm)
        print("mean |err|  model       : %.2f" % mm)
        print("model vs persistence     : %.4f  (lower is better)" % (mm / pm))
        print("cases won                : %d/%d" % (wins, len(rows)))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
