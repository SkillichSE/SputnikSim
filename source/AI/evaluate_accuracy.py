# evaluate_accuracy.py
# Evaluates model accuracy using both the supervised classifier and reconstruction error.
#
# Usage:
#   python evaluate_accuracy.py
#   python evaluate_accuracy.py --model model/tle_model_best.pth --normalizer model/normalizer.pkl

import sys
import os
import argparse
import requests
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parse_tle import parse_tle, get_orbit_type, TLEParseError
from train_model import (
    TLEAnalyzer, TLENormalizer, INPUT_DIM,
    make_full_vector, _make_physics_features,
    BASE_FEATURE_DIM, PHYSICS_FEATURE_DIM,
)

# p95 reconstruction error per orbit type (from train_model.py evaluation)
ORBIT_THRESHOLDS = {
    "LEO":       0.606,
    "LEO_POLAR": 0.753,
    "SSO":       0.709,
    "MEO":       0.733,
    "GEO":       1.436,
    "GSO":       1.386,
    "HEO":       3.222,
    "OTHER":     1.977,
}

CELESTRAK_NORMAL = [
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",   "active satellites"),
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle", "space stations"),
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",  "GPS"),
]
CELESTRAK_ANOMALY = [
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle", "COSMOS-1408 debris"),
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle",  "Fengyun-1C debris"),
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle",  "Iridium-33 debris"),
    ("https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle", "Cosmos-2251 debris"),
]


def fetch_tle(url, label, max_records=400):
    print(f"  Downloading: {label}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        lines = [l.strip() for l in r.text.splitlines() if l.strip()]
        records = []
        for i in range(0, len(lines) - 2, 3):
            if len(records) >= max_records:
                break
            records.append((lines[i], lines[i+1], lines[i+2]))
        print(f"    Got {len(records)} records")
        return records
    except Exception as e:
        print(f"    FAILED: {e}")
        return []


def load_model(model_path, normalizer_path, device):
    model = TLEAnalyzer(input_dim=INPUT_DIM)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    normalizer = TLENormalizer()
    normalizer.load(normalizer_path)
    return model, normalizer


def predict_batch(model, normalizer, records, device):
    results = []
    for name, l1, l2 in records:
        try:
            vector, tle_struct = parse_tle(name, l1, l2)
            orbit_type = get_orbit_type(
                tle_struct["mean_motion"],
                tle_struct["inclination"],
                tle_struct["eccentricity"],
            )
            threshold = ORBIT_THRESHOLDS.get(orbit_type, 0.15)
            physics   = _make_physics_features(tle_struct)
            aug       = make_full_vector(vector, orbit_type, physics)
            vec_norm  = normalizer.transform(np.array(aug))
            t         = torch.tensor(vec_norm, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                recon, latent = model(t)
                recon_err     = torch.mean((t - recon) ** 2).item()
                cls_score     = model.classifier(latent).item()

            pred_cls   = 1 if cls_score >= 0.5 else 0
            pred_recon = 1 if recon_err > threshold else 0
            # combined: classifier is primary (weight 2), recon is secondary (weight 1)
            # anomaly only if weighted score >= 1.5 (i.e. cls alone, or both agree)
            vote = (2 if pred_cls else 0) + (1 if pred_recon else 0)
            pred_combined = 1 if vote >= 2 else 0

            results.append({
                "name":         name,
                "orbit_type":   orbit_type,
                "recon_error":  recon_err,
                "cls_score":    cls_score,
                "threshold":    threshold,
                "pred_cls":     pred_cls,
                "pred_recon":   pred_recon,
                "pred_combined": pred_combined,
            })
        except Exception:
            pass
    return results


def print_metrics(label, y_true, y_pred):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n  [{label}]")
    print(f"  Accuracy:  {acc*100:.1f}%   Precision: {prec*100:.1f}%   "
          f"Recall: {rec*100:.1f}%   F1: {f1*100:.1f}%")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")
    return acc, f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate anomaly detection accuracy")
    parser.add_argument("--model",       "-m", default="model/tle_model_best.pth")
    parser.add_argument("--normalizer",  "-n", default="model/normalizer.pkl")
    parser.add_argument("--cpu",               action="store_true")
    parser.add_argument("--max-samples",       type=int, default=300)
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        return 1
    if not os.path.exists(args.normalizer):
        print(f"ERROR: Normalizer not found: {args.normalizer}")
        return 1

    print(f"\nLoading model...")
    model, normalizer = load_model(args.model, args.normalizer, device)

    print("\nDownloading NORMAL satellites...")
    normal_records = []
    for url, label in CELESTRAK_NORMAL:
        normal_records += fetch_tle(url, label, max_records=args.max_samples // len(CELESTRAK_NORMAL) + 50)
    normal_records = normal_records[:args.max_samples]

    print("\nDownloading ANOMALOUS objects (debris)...")
    anomaly_records = []
    for url, label in CELESTRAK_ANOMALY:
        anomaly_records += fetch_tle(url, label, max_records=args.max_samples // len(CELESTRAK_ANOMALY) + 50)
    anomaly_records = anomaly_records[:args.max_samples]

    if not normal_records or not anomaly_records:
        print("ERROR: Not enough data. Check internet connection.")
        return 1

    print(f"\nRunning inference on {len(normal_records)} normal + {len(anomaly_records)} anomaly records...")
    normal_results  = predict_batch(model, normalizer, normal_records,  device)
    anomaly_results = predict_batch(model, normalizer, anomaly_records, device)

    y_true = [0] * len(normal_results) + [1] * len(anomaly_results)

    print("\n" + "=" * 55)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ ТОЧНОСТИ")
    print("=" * 55)
    print(f"Нормальных спутников:  {len(normal_results)}")
    print(f"Аномальных объектов:   {len(anomaly_results)}")

    y_cls      = [r["pred_cls"]      for r in normal_results + anomaly_results]
    y_recon    = [r["pred_recon"]    for r in normal_results + anomaly_results]
    y_combined = [r["pred_combined"] for r in normal_results + anomaly_results]

    print_metrics("Классификатор (новый)", y_true, y_cls)
    print_metrics("Реконструкция (старый)", y_true, y_recon)
    acc_comb, f1_comb = print_metrics("Комбинированный", y_true, y_combined)

    print("\n" + "=" * 55)
    print(f"ИТОГ: Точность (комбинированная) = {acc_comb*100:.1f}%  F1 = {f1_comb*100:.1f}%")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    exit(main())