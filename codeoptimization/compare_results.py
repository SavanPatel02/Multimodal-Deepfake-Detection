"""
Comparison table across all 10 trained models.
Run this after training all models with train_pair.py.

Usage:
    python compare_results.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from logger_setup import setup_logger

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

MODEL_INFO = {
    "M1":  ("Pair 1", "ResNet18",       "Custom CNN",       "Concat"),
    "M2":  ("Pair 1", "ResNet50",        "Custom CNN",       "Concat"),
    "M3":  ("Pair 2", "ResNet18",        "CNN Audio",        "Concat"),
    "M4":  ("Pair 2", "ResNet18",        "LSTM Audio",       "Concat"),
    "M5":  ("Pair 3", "MobileNetV3-S",   "Lightweight CNN",  "Concat"),
    "M6":  ("Pair 3", "EfficientNet-B0", "Custom CNN",       "Concat"),
    "M7":  ("Pair 4", "ResNet18",        "Custom CNN",       "Concat"),
    "M8":  ("Pair 4", "ResNet18",        "Custom CNN",       "Cross-Attention"),
    "M9":  ("Pair 5", "ResNet18",        "— (Visual Only)",  "—"),
    "M10": ("Pair 5", "— (Audio Only)",  "Custom CNN",       "—"),
}

PAIR_QUESTIONS = {
    "Pair 1": "Does deeper visual backbone help?",
    "Pair 2": "CNN vs LSTM for audio?",
    "Pair 3": "Can lightweight models compete?",
    "Pair 4": "Does cross-attention beat concat?",
    "Pair 5": "Is multimodal better than unimodal?",
}


def load_result(model_name):
    path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    log = setup_logger("compare_results")

    log.info("\n" + "=" * 95)
    log.info("  MULTIMODAL DEEPFAKE DETECTION — MODEL COMPARISON")
    log.info("=" * 95)

    header = f"{'Model':<6} {'Pair':<8} {'Visual':<18} {'Audio':<18} {'Fusion':<16} {'Val Acc':>8} {'Val F1':>8} {'Params':>10}"
    log.info(f"\n{header}")
    log.info("-" * 95)

    all_results = []

    for model_name, (pair, visual, audio, fusion) in MODEL_INFO.items():
        result = load_result(model_name)

        if result:
            acc    = f"{result['best_val_acc']:.2f}%"
            f1     = f"{result['best_val_f1']:.4f}"
            params = f"{result['parameters']:,}"
            all_results.append((model_name, result['best_val_acc'], result['best_val_f1']))
        else:
            acc, f1, params = "—", "—", "—"

        log.info(f"{model_name:<6} {pair:<8} {visual:<18} {audio:<18} {fusion:<16} {acc:>8} {f1:>8} {params:>10}")

    log.info("=" * 95)

    # Per-pair winner
    log.info("\n  PER-PAIR ANALYSIS")
    log.info("-" * 95)

    pairs = {}
    for model_name, (pair, *_) in MODEL_INFO.items():
        pairs.setdefault(pair, []).append(model_name)

    for pair, models in pairs.items():
        log.info(f"\n  {pair} — {PAIR_QUESTIONS[pair]}")
        for m in models:
            result = load_result(m)
            info = MODEL_INFO[m]
            if result:
                log.info(f"    {m} ({info[1]} + {info[2]}): {result['best_val_acc']:.2f}% acc | F1: {result['best_val_f1']:.4f}")
            else:
                log.info(f"    {m} ({info[1]} + {info[2]}): not yet trained")

        trained = [(m, load_result(m)) for m in models if load_result(m)]
        if len(trained) == 2:
            winner = max(trained, key=lambda x: x[1]['best_val_acc'])
            loser  = min(trained, key=lambda x: x[1]['best_val_acc'])
            diff   = winner[1]['best_val_acc'] - loser[1]['best_val_acc']
            log.info(f"    → Winner: {winner[0]} (+{diff:.2f}%)")

    # Overall ranking
    if all_results:
        log.info("\n" + "=" * 95)
        log.info("  OVERALL RANKING (by Val Accuracy)")
        log.info("-" * 95)
        ranked = sorted(all_results, key=lambda x: x[1], reverse=True)
        for rank, (model_name, acc, f1) in enumerate(ranked, 1):
            info = MODEL_INFO[model_name]
            log.info(f"  #{rank:>2}  {model_name:<5} | {acc:.2f}% acc | F1: {f1:.4f} | {info[1]} + {info[2]}")

    log.info("\n" + "=" * 95 + "\n")


if __name__ == "__main__":
    main()
