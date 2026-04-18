"""
GPU Batch Size Finder — Auto-updates train_pair.py BATCH_SIZES
Targets 2-3 GB VRAM headroom out of 16 GB (uses up to 13 GB max).

Usage:
    python codeoptimization/find_batch_size.py
"""

import torch
import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("ERROR: No CUDA GPU detected. This script requires a GPU.")
    sys.exit(1)

total_vram   = torch.cuda.get_device_properties(0).total_memory / 1024**3
HEADROOM_GB  = 2.5                          # keep this much VRAM free
TARGET_GB    = total_vram - HEADROOM_GB     # e.g. 16 - 2.5 = 13.5 GB max
TARGET_FRAC  = TARGET_GB / total_vram       # fraction of total VRAM to target

# Prevent spilling into shared system RAM (would give falsely large batches)
torch.cuda.set_per_process_memory_fraction(TARGET_FRAC)

print(f"\n{'='*60}")
print(f"  GPU      : {torch.cuda.get_device_name(0)}")
print(f"  VRAM     : {total_vram:.1f} GB total")
print(f"  Target   : {TARGET_GB:.1f} GB  (headroom: {HEADROOM_GB} GB)")
print(f"{'='*60}\n")


# ── Model registry ────────────────────────────────────────────────────────────
def get_model(name):
    name = name.upper()
    if name == "M1":
        from pair1_models import M1_ResNet18_CNN;         return M1_ResNet18_CNN()
    elif name == "M2":
        from pair1_models import M2_ResNet50_CNN;         return M2_ResNet50_CNN()
    elif name == "M3":
        from pair2_models import M3_ResNet18_CNNAudio;    return M3_ResNet18_CNNAudio()
    elif name == "M4":
        from pair2_models import M4_ResNet18_LSTMAudio;   return M4_ResNet18_LSTMAudio()
    elif name == "M5":
        from pair3_models import M5_MobileNetV3_LightCNN; return M5_MobileNetV3_LightCNN()
    elif name == "M6":
        from pair3_models import M6_EfficientNetB0_CNN;   return M6_EfficientNetB0_CNN()
    elif name == "M7":
        from pair4_models import M7_ConcatFusion;         return M7_ConcatFusion()
    elif name == "M8":
        from pair4_models import M8_AttentionFusion;      return M8_AttentionFusion()
    elif name == "M9":
        from pair5_models import M9_VisualOnly;           return M9_VisualOnly()
    elif name == "M10":
        from pair5_models import M10_AudioOnly;           return M10_AudioOnly()


# ── Dummy inputs ──────────────────────────────────────────────────────────────
def make_inputs(batch_size):
    frames = torch.randn(batch_size, 16, 3, 224, 224, device=DEVICE)
    mel    = torch.randn(batch_size, 1, 128, 126,      device=DEVICE)
    labels = torch.zeros(batch_size, dtype=torch.long, device=DEVICE)
    return frames, mel, labels


# ── Forward + backward probe ──────────────────────────────────────────────────
def probe(model, batch_size):
    """Returns (vram_peak_GB, throughput) or raises OOM."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler    = torch.cuda.amp.GradScaler()

    frames, mel, labels = make_inputs(batch_size)

    import time

    # Warm-up pass
    with torch.amp.autocast(device_type="cuda"):
        out  = model(frames, mel)
        loss = criterion(out, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed pass
    t0 = time.perf_counter()
    with torch.amp.autocast(device_type="cuda"):
        out  = model(frames, mel)
        loss = criterion(out, labels)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak_gb    = torch.cuda.max_memory_allocated() / 1024**3
    throughput = batch_size / elapsed
    return peak_gb, throughput


# ── Find batch size targeting headroom ────────────────────────────────────────
def find_batch(model_name):
    """
    Binary search for the largest batch size that stays within TARGET_GB.
    Always rounds to nearest multiple of 8 (tensor core alignment).
    Returns the safe batch size.
    """
    model  = get_model(model_name).to(DEVICE)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  {model_name}  ({params:,} params)")

    lo, hi   = 8, 8
    last_ok  = None   # (batch_size, vram_gb, throughput)

    # Phase 1 — double until we exceed TARGET_GB or OOM
    while True:
        try:
            vram, tput = probe(model, hi)
            if vram <= TARGET_GB:
                last_ok = (hi, vram, tput)
                if hi >= 1024:
                    break
                hi *= 2
            else:
                # Over target — binary search between lo and hi
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            break

    if last_ok is None:
        print(f"    OOM even at batch=8. Using batch=8 as fallback.\n")
        del model
        torch.cuda.empty_cache()
        return 8

    lo = last_ok[0]

    # Phase 2 — binary search between lo and hi
    while lo + 8 <= hi:
        mid = ((lo + hi) // 2 // 8) * 8   # keep multiple of 8
        if mid <= lo:
            break
        try:
            vram, tput = probe(model, mid)
            if vram <= TARGET_GB:
                last_ok = (mid, vram, tput)
                lo = mid
            else:
                hi = mid
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid

    bs, vram, tput = last_ok
    headroom = total_vram - vram

    print(f"    Batch size : {bs}")
    print(f"    VRAM used  : {vram:.2f} GB / {total_vram:.1f} GB  "
          f"({vram/total_vram*100:.0f}%)  —  {headroom:.1f} GB free")
    print(f"    Throughput : {tput:.1f} samples/sec")
    print()

    del model
    torch.cuda.empty_cache()
    return bs


# ── Auto-update BATCH_SIZES in train_pair.py ──────────────────────────────────
def update_train_pair(new_sizes: dict):
    train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pair.py")

    with open(train_path, "r") as f:
        src = f.read()

    # Build replacement block
    lines = ["BATCH_SIZES = {\n"]
    for k, v in new_sizes.items():
        lines.append(f'    "{k}": {v:>4},\n')
    lines.append("}")
    new_block = "".join(lines)

    # Replace existing BATCH_SIZES dict
    updated = re.sub(
        r'BATCH_SIZES\s*=\s*\{[^}]*\}',
        new_block,
        src,
        flags=re.DOTALL
    )

    if updated == src:
        print("  WARNING: Could not locate BATCH_SIZES in train_pair.py — no changes made.")
        return

    with open(train_path, "w") as f:
        f.write(updated)

    print(f"  train_pair.py updated with new BATCH_SIZES.")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    models = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]

    print("Testing forward+backward pass with AMP...\n")
    print(f"{'─'*60}")

    results = {}
    for m in models:
        try:
            results[m] = find_batch(m)
        except Exception as e:
            print(f"  {m}: unexpected error — {e}. Using batch=8.\n")
            results[m] = 8
        torch.cuda.empty_cache()

    # Summary table
    print(f"{'='*60}")
    print(f"  {'Model':<6}  {'Batch':>6}  {'Est. VRAM headroom':>20}")
    print(f"  {'─'*50}")
    for m, bs in results.items():
        print(f"  {m:<6}  {bs:>6}")
    print(f"{'='*60}\n")

    # Auto-update train_pair.py
    update_train_pair(results)
    print("\n  Done. Rerun training with: python codeoptimization/train_pair.py --model M1\n")
