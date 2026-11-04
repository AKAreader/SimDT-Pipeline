#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1: Anti-UAV410 Representative Diagnostic Subset Architect & Builder
=========================================================================
Strictly Read-Only Upstream Data Sources:
  - E:\\PHD\\YB\\Anti_yb\\annos\\val\\att\\<seq>.txt (10 official binary attributes)
  - E:\\PHD\\YB\\dataset\\Anti-UAV410\\Anti-UAV\\val\\<seq>\\IR_label.json (per-frame GT)
  - E:\\PHD\\YB\\yb\\research_20261103\\failure_diagnosis\\BASELINE_FULLVAL90_STATS.json

Deliverables Written (Strictly inside SimDT workspace):
  - phase1_diagnostic_subset/DIAGNOSTIC_SUBSET_MANIFEST.json
  - datasets/diagnostic_splits/val90_full.json
  - datasets/diagnostic_splits/discovery_set.json
  - datasets/diagnostic_splits/holdout_verification.json

Core Principles:
  1. Zero Tracker-Score Cherry-Picking: Selection and partitioning use ONLY
     ground-truth annotations, official attribute labels, and source recording
     session provenance (recording_group).
  2. Strict Recording-Group Disjointness: All sibling clips originating from
     the same source recording session are assigned to the same split tier
     (25 Discovery Recording Groups = 54 sequences; 18 Holdout Recording Groups
     = 36 sequences; 0 shared recording groups).
  3. Two-Tier Hierarchical Consistency:
     - Tier 1 (Full Val90 Split): 54 Discovery : 36 Holdout sequences.
     - Tier 2 (Core Diagnostic Subset 40): 24 Discovery : 16 Holdout sequences,
       where Core Discovery (24) is a strict subset of Full Discovery (54) and
       Core Holdout (16) is a strict subset of Full Holdout (36).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

ATTRIBUTE_NAMES: List[str] = [
    "TC",   # Thermal Crossover
    "OV",   # Out-of-View
    "SV",   # Scale Variation
    "FM",   # Fast Motion
    "OC",   # Occlusion
    "DBC",  # Dynamic Background Clutter
    "TS",   # Tiny Size
    "SS",   # Small Size
    "MS",   # Medium Size
    "NS",   # Normal Size
]

DEFAULT_ATT_DIR = Path(r"E:\PHD\YB\Anti_yb\annos\val\att")
DEFAULT_VAL_ROOT = Path(r"E:\PHD\YB\dataset\Anti-UAV410\Anti-UAV\val")
DEFAULT_STATS_JSON = Path(
    r"E:\PHD\YB\yb\research_20261103\failure_diagnosis\BASELINE_FULLVAL90_STATS.json"
)

# Frozen Recording-Group Partition (25 Discovery Groups : 18 Holdout Groups)
DISCOVERY_GROUPS_25: List[str] = [
    "03_7951_1500-2999",
    "20190925_130434_1",
    "20190925_131530_1",
    "20190925_141417_1",
    "20190925_152412_1",
    "20190925_194211_1",
    "20190925_205804_1",
    "20190925_210802_1",
    "20190925_222534_1",
    "20190926_095902_1",
    "20190926_103046_1",
    "20190926_130341_1",
    "20190926_133516_1",
    "20190926_143632_1",
    "20190926_183400_1",
    "20190926_200510_1",
    "3700000000002_105103",
    "3700000000002_105837",
    "3700000000002_110743",
    "3700000000002_153139",
    "new28_train-new",
    "new29_train-new",
    "new36_train-new",
    "new40_train-new",
    "new8_train-new",
]

HOLDOUT_GROUPS_18: List[str] = [
    "20190925_133630_1",
    "20190925_140917_1",
    "20190925_143900_1",
    "20190925_183946_1",
    "20190925_200320_1",
    "20190925_213001_1",
    "20190926_141816_1",
    "20190926_142435_1",
    "20190926_144550_1",
    "20190926_183941_1",
    "20190926_195921_1",
    "3700000000002_091732",
    "3700000000002_105009",
    "3700000000002_110352",
    "3700000000002_111356",
    "3700000000002_143252",
    "3700000000002_162623",
    "new30_train-new",
]

# Frozen Core Diagnostic Subset 40 (24 Discovery : 16 Holdout)
CORE_DISCOVERY_24: List[str] = [
    "03_7951_1500-2999",
    "20190925_130434_1_8",
    "20190925_131530_1_3",
    "20190925_141417_1_1",
    "20190925_152412_1_3",
    "20190925_194211_1_3",
    "20190925_205804_1_5",
    "20190925_205804_1_6",
    "20190925_210802_1_3",
    "20190925_222534_1_5",
    "20190926_095902_1_3",
    "20190926_095902_1_8",
    "20190926_103046_1_2",
    "20190926_143632_1_8",
    "20190926_183400_1_4",
    "20190926_200510_1_2",
    "3700000000002_105103_8",
    "3700000000002_105837_2",
    "3700000000002_110743_1",
    "3700000000002_153139_1",
    "new28_train-new",
    "new29_train-new",
    "new36_train-new",
    "new40_train-new",
]

CORE_HOLDOUT_16: List[str] = [
    "20190925_140917_1_3",
    "20190925_143900_1_1",
    "20190925_183946_1_1",
    "20190925_200320_1_4",
    "20190925_213001_1_1",
    "20190926_141816_1_9",
    "20190926_142435_1_9",
    "20190926_144550_1_1",
    "20190926_183941_1_3",
    "20190926_195921_1_8",
    "3700000000002_091732_1",
    "3700000000002_105009_1",
    "3700000000002_110352_1",
    "3700000000002_143252_1",
    "3700000000002_162623_2",
    "new30_train-new",
]


def infer_recording_group(seq_name: str) -> str:
    """Extract canonical source recording group ID from sequence name."""
    if seq_name.startswith("201909"):
        parts = seq_name.split("_")
        return "_".join(parts[:3])  # e.g., 20190925_130434_1
    if seq_name.startswith("3700000000002_"):
        parts = seq_name.split("_")
        return "_".join(parts[:2])  # e.g., 3700000000002_105103
    return seq_name


def percentile(sorted_vals: List[float], q: float) -> float:
    """Compute linear-interpolated percentile q in [0, 100] from sorted list."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_kinematic_metrics_from_ir_label(
    label_path: Path, frame_w: float = 640.0, frame_h: float = 512.0
) -> Dict[str, Any]:
    """Compute objective physical & kinematic metrics from IR_label.json."""
    raw_bytes = label_path.read_bytes()
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    md5_hex = hashlib.md5(raw_bytes).hexdigest()
    data = json.loads(raw_bytes.decode("utf-8"))

    exist_list: List[int] = [int(x) for x in data.get("exist", [])]
    gt_rects: List[List[float]] = data.get("gt_rect", [])
    n_frames = len(exist_list)

    absent_frames = sum(1 for e in exist_list if e == 0)
    reentry_count = 0
    for i in range(1, n_frames):
        if exist_list[i - 1] == 0 and exist_list[i] == 1:
            reentry_count += 1

    sizes: List[float] = []
    areas: List[float] = []
    aspect_ratios: List[float] = []
    centers: List[Tuple[int, float, float]] = []
    border_frames = 0
    valid_present = 0
    invalid_present = 0

    for idx, e in enumerate(exist_list):
        if e == 0:
            continue
        rect = gt_rects[idx] if idx < len(gt_rects) else [0, 0, 0, 0]
        if len(rect) < 4 or rect[2] <= 0 or rect[3] <= 0:
            invalid_present += 1
            continue
        x, y, w, h = float(rect[0]), float(rect[1]), float(rect[2]), float(rect[3])
        valid_present += 1
        area = w * h
        size = math.sqrt(area)
        sizes.append(size)
        areas.append(area)
        aspect_ratios.append(w / max(h, 1e-6))
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        centers.append((idx, cx, cy))
        if x <= 8.0 or y <= 8.0 or (x + w) >= (frame_w - 8.0) or (y + h) >= (frame_h - 8.0):
            border_frames += 1

    speeds: List[float] = []
    for i in range(1, len(centers)):
        t0, x0, y0 = centers[i - 1]
        t1, x1, y1 = centers[i]
        if t1 - t0 == 1:
            speeds.append(math.hypot(x1 - x0, y1 - y0))

    accels: List[float] = []
    for i in range(1, len(speeds)):
        accels.append(abs(speeds[i] - speeds[i - 1]))

    ar_changes: List[float] = []
    for i in range(1, len(aspect_ratios)):
        ar_changes.append(abs(aspect_ratios[i] - aspect_ratios[i - 1]) / max(aspect_ratios[i - 1], 1e-6))

    sizes_sorted = sorted(sizes)
    speeds_sorted = sorted(speeds)
    accels_sorted = sorted(accels)
    areas_sorted = sorted(areas)

    tiny_lt16 = sum(1 for s in sizes if s < 16.0)
    tiny_lt10 = sum(1 for s in sizes if s < 10.0)
    denom = max(valid_present, 1)

    scale_max_min_ratio = (
        float(areas_sorted[-1] / max(areas_sorted[0], 1e-6)) if areas_sorted else 1.0
    )

    return {
        "frames": n_frames,
        "sa_denominator": n_frames - invalid_present,
        "present_valid_frames": valid_present,
        "absent_frames": absent_frames,
        "absent_rate": round(absent_frames / max(n_frames, 1), 6),
        "reentry_count": reentry_count,
        "size_mean": round(sum(sizes) / denom, 4),
        "size_p50": round(percentile(sizes_sorted, 50.0), 4),
        "size_p10": round(percentile(sizes_sorted, 10.0), 4),
        "tiny_lt16_rate": round(tiny_lt16 / denom, 6),
        "tiny_lt10_rate": round(tiny_lt10 / denom, 6),
        "speed_mean": round(sum(speeds) / max(len(speeds), 1), 4),
        "speed_p95": round(percentile(speeds_sorted, 95.0), 4),
        "accel_p95": round(percentile(accels_sorted, 95.0), 4),
        "scale_max_min_ratio": round(scale_max_min_ratio, 4),
        "aspect_ratio_change_rate": round(sum(ar_changes) / max(len(ar_changes), 1), 6),
        "border_proximity_rate": round(border_frames / denom, 6),
        "label_sha256": sha256_hex,
        "label_md5": md5_hex,
    }


def read_official_attributes(att_path: Path) -> Dict[str, int]:
    """Parse 10 binary flags [TC, OV, SV, FM, OC, DBC, TS, SS, MS, NS]."""
    raw = att_path.read_text(encoding="utf-8").strip()
    tokens = [int(tok.strip()) for tok in raw.replace(",", " ").split() if tok.strip()]
    if len(tokens) != 10:
        raise ValueError(f"Expected 10 attribute flags in {att_path}, got {len(tokens)}")
    return {name: int(val) for name, val in zip(ATTRIBUTE_NAMES, tokens)}


def two_sample_ks_statistic(sample_a: List[float], sample_b: List[float]) -> Tuple[float, float]:
    """Compute exact two-sample Kolmogorov-Smirnov D statistic and asymptotic p-value."""
    if not sample_a or not sample_b:
        return 0.0, 1.0
    a_sorted = sorted(sample_a)
    b_sorted = sorted(sample_b)
    n1 = len(a_sorted)
    n2 = len(b_sorted)
    all_Points = sorted(set(a_sorted + b_sorted))
    d_max = 0.0
    i = 0
    j = 0
    for pt in all_Points:
        while i < n1 and a_sorted[i] <= pt:
            i += 1
        while j < n2 and b_sorted[j] <= pt:
            j += 1
        cdf1 = i / n1
        cdf2 = j / n2
        d = abs(cdf1 - cdf2)
        if d > d_max:
            d_max = d
    en = math.sqrt((n1 * n2) / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d_max
    if lam <= 1e-6:
        return 0.0, 1.0
    p_val = 0.0
    for k in range(1, 25):
        p_val += 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k * lam) ** 2)
    p_val = max(0.0, min(1.0, p_val))
    return round(d_max, 6), round(p_val, 6)


def verify_partition_invariants() -> Dict[str, Any]:
    """Verify strict group disjointness and hierarchy invariants."""
    disc_groups = set(DISCOVERY_GROUPS_25)
    hold_groups = set(HOLDOUT_GROUPS_18)
    overlap_groups = disc_groups.intersection(hold_groups)
    assert len(overlap_groups) == 0, f"Group leakage detected: {overlap_groups}"
    assert len(disc_groups) + len(hold_groups) == 43, "Must partition all 43 recording groups"

    core_disc_groups = {infer_recording_group(s) for s in CORE_DISCOVERY_24}
    core_hold_groups = {infer_recording_group(s) for s in CORE_HOLDOUT_16}
    assert core_disc_groups.issubset(disc_groups), "Core Discovery must be subset of Discovery groups"
    assert core_hold_groups.issubset(hold_groups), "Core Holdout must be subset of Holdout groups"
    assert len(core_disc_groups.intersection(core_hold_groups)) == 0, "Zero group overlap in Core 40"

    return {
        "total_recording_groups": 43,
        "discovery_recording_groups": len(disc_groups),
        "holdout_recording_groups": len(hold_groups),
        "shared_recording_groups": len(overlap_groups),
        "core_discovery_sequences": len(CORE_DISCOVERY_24),
        "core_holdout_sequences": len(CORE_HOLDOUT_16),
        "core_discovery_unique_groups": len(core_disc_groups),
        "core_holdout_unique_groups": len(core_hold_groups),
        "invariants_passed": True,
    }


if __name__ == "__main__":
    report = verify_partition_invariants()
    print(json.dumps(report, indent=2, ensure_ascii=False))
