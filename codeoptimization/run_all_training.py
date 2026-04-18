"""
Run all 10 models in optimal order — fastest first, slowest last.

Order: M10 → M5 → M8 → M3 → M4 → M7 → M9 → M1 → M6 → M2

Skips models that already have a completed results JSON.
Logs overall progress and time per model.

Usage:
    python codeoptimization/run_all_training.py

    # Train specific models only:
    python codeoptimization/run_all_training.py --models M8 M5 M10

    # Force retrain even if results exist:
    python codeoptimization/run_all_training.py --force
"""

import os
import sys
import json
import time
import argparse
import subprocess

BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR      = os.path.join(BASE_DIR, "codeoptimization", "results")
RESULTS_DIR_SUB  = os.path.join(BASE_DIR, "codeoptimization", "results", "subset")
TRAIN_SCRIPT     = os.path.join(BASE_DIR, "codeoptimization", "train_pair.py")

# Fastest → slowest based on throughput benchmarks
ORDER = ["M10", "M5", "M8", "M3", "M4", "M7", "M9", "M1", "M6", "M2"]


def already_done(model_name, subset=False):
    """Returns True if a completed results JSON exists for this model."""
    results_dir = RESULTS_DIR_SUB if subset else RESULTS_DIR
    path = os.path.join(results_dir, f"{model_name}.json")
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        return "best_val_acc" in data and "history" in data and len(data["history"]) > 0
    except Exception:
        return False


def fmt_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def print_separator(char="=", width=60):
    print(char * width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ORDER,
                        help="Models to train (default: all in optimal order)")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if results already exist")
    parser.add_argument("--subset", action="store_true",
                        help="Use subset CSVs for quick model comparison (~5-8 min/epoch)")
    args = parser.parse_args()

    # Validate model names
    valid = set(ORDER)
    for m in args.models:
        if m.upper() not in valid:
            print(f"ERROR: Unknown model '{m}'. Valid: {', '.join(ORDER)}")
            sys.exit(1)

    # Preserve optimal order even if user specified --models
    models_to_run = [m for m in ORDER if m in [x.upper() for x in args.models]]

    results_dir = RESULTS_DIR_SUB if args.subset else RESULTS_DIR
    os.makedirs(results_dir, exist_ok=True)

    print_separator()
    print(f"  TRAINING RUN — {len(models_to_run)} models{'  [SUBSET MODE]' if args.subset else ''}")
    print(f"  Order: {' → '.join(models_to_run)}")
    if args.subset:
        print(f"  Results → {results_dir}")
    print_separator()
    print()

    session_start = time.time()
    completed     = []
    skipped       = []
    failed        = []

    for i, model in enumerate(models_to_run, 1):
        print_separator("-")
        print(f"  [{i}/{len(models_to_run)}]  {model}")

        if not args.force and already_done(model, subset=args.subset):
            print(f"  SKIP — results already exist ({model}.json)")
            skipped.append(model)
            print()
            continue

        # Clean stale resume checkpoints when forcing retrain
        if args.force:
            resume_path = os.path.join(results_dir, f"{model}_resume.pth")
            if os.path.exists(resume_path):
                os.remove(resume_path)

        print(f"  Starting training...")
        print_separator("-")

        cmd = [sys.executable, TRAIN_SCRIPT, "--model", model]
        if args.subset:
            cmd.append("--subset")

        t0 = time.time()
        result = subprocess.run(cmd, cwd=BASE_DIR)
        elapsed = time.time() - t0

        if result.returncode == 0:
            completed.append((model, elapsed))
            print_separator("-")
            print(f"  {model} DONE — {fmt_time(elapsed)}")

            # Print best result from JSON
            results_path = os.path.join(results_dir, f"{model}.json")
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    print(f"  Best val acc : {data['best_val_acc']:.2f}%")
                    print(f"  Best val F1  : {data['best_val_f1']:.4f}")
                    print(f"  Epochs run   : {data['epochs_run']}")
                except Exception:
                    pass
        else:
            failed.append(model)
            print_separator("-")
            print(f"  {model} FAILED (exit code {result.returncode})")

        print()

    # Final summary
    total_time = time.time() - session_start
    print_separator()
    print(f"  SESSION COMPLETE — {fmt_time(total_time)}")
    print_separator()

    if completed:
        print(f"\n  Completed ({len(completed)}):")
        for model, t in completed:
            results_path = os.path.join(results_dir, f"{model}.json")
            acc_str = ""
            if os.path.exists(results_path):
                try:
                    with open(results_path) as f:
                        data = json.load(f)
                    acc_str = f"  val acc {data['best_val_acc']:.2f}%  F1 {data['best_val_f1']:.4f}"
                except Exception:
                    pass
            print(f"    {model:<6} — {fmt_time(t):<12}{acc_str}")

    if skipped:
        print(f"\n  Skipped (already done): {', '.join(skipped)}")
        print(f"  Use --force to retrain them.")

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")
        print(f"  Check logs in codeoptimization/logs/")

    # ── Ranking table across ALL result JSONs in the results dir ──────────────
    print(f"\n{'='*60}")
    print(f"  MODEL RANKING  (by val F1, then val acc)")
    print(f"  {'Model':<6}  {'Val Acc':>8}  {'Val F1':>8}  {'Epochs':>7}  {'Params':>12}")
    print(f"  {'─'*54}")
    ranking = []
    for m in ORDER:
        rp = os.path.join(results_dir, f"{m}.json")
        if os.path.exists(rp):
            try:
                with open(rp) as f:
                    d = json.load(f)
                ranking.append((m, d["best_val_acc"], d["best_val_f1"],
                                d["epochs_run"], d.get("parameters", 0)))
            except Exception:
                pass
    ranking.sort(key=lambda x: (x[2], x[1]), reverse=True)
    for rank, (m, acc, f1, ep, params) in enumerate(ranking, 1):
        star = " ★" if rank == 1 else ""
        print(f"  {m:<6}  {acc:>7.2f}%  {f1:>8.4f}  {ep:>7}  {params:>12,}{star}")
    if not ranking:
        print(f"  (no completed results yet)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
