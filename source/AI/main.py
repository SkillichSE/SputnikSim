# main.py (AI/)
import torch
import numpy as np
from train_model import TLEAnalyzer, TLENormalizer, INPUT_DIM, ALL_ORBIT_TYPES, _make_augmented_vector, FEATURE_NAMES
from parse_tle import parse_tle, TLEParseError, get_orbit_type, classify_orbit
from generate import generate_detailed_text_with_values, generate_summary_text
import os
import argparse
from datetime import datetime


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

# normal BSTAR limits per orbit type
_BSTAR_NORMAL = {
    "LEO": 0.001, "LEO_POLAR": 0.001, "SSO": 0.001,
    "MEO": 0.0001, "GEO": 0.0001, "GSO": 0.0001,
    "HEO": 0.001, "OTHER": 0.001,
}
_DV_NORMAL_MAX = 50.0  # m/s
_OOD_Z_WARN   = 3.0   # z-score threshold


def _physical_status(tle_struct, orbit_type):
    # returns True if physical params are in normal range
    bstar     = abs(tle_struct.get("bstar", 0))
    bstar_ok  = bstar <= _BSTAR_NORMAL.get(orbit_type, 0.001)
    mm_dot_ok = abs(tle_struct.get("mean_motion_dot", 0)) <= 0.0002
    return bstar_ok and mm_dot_ok


def _station_keeping_maneuver(tle_struct, orbit_type):
    # detects station-keeping maneuver: high dn/dt + normal BSTAR
    bstar  = abs(tle_struct.get("bstar", 0))
    mm_dot = abs(tle_struct.get("mean_motion_dot", 0))
    if mm_dot > 0.0001 and bstar <= _BSTAR_NORMAL.get(orbit_type, 0.001):
        return True, (
            f"Признак манёвра поддержания высоты: dn/dt={tle_struct['mean_motion_dot']:.2e} "
            f"при нормальном BSTAR={bstar:.2e}. Это штатное орбитальное манёврирование, "
            f"а не аномалия атмосферного торможения."
        )
    return False, ""


class TLEAnalysisSystem:
    """Multi-orbit TLE anomaly detection system."""

    def __init__(self, model_path="model/tle_model.pth",
                 normalizer_path="model/normalizer.pkl",
                 device="cpu"):
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}\nRun train_model.py first.")

        self.model = TLEAnalyzer(input_dim=INPUT_DIM)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        if not os.path.exists(normalizer_path):
            raise FileNotFoundError(f"Normalizer not found: {normalizer_path}\nRun train_model.py first.")

        self.normalizer = TLENormalizer()
        self.normalizer.load(normalizer_path)

    def _threshold_for(self, orbit_type):
        return ORBIT_THRESHOLDS.get(orbit_type, 0.15)

    def analyze_single_tle(self, name_line, line1, line2, verbose=True):
        try:
            vector, tle_struct = parse_tle(name_line, line1, line2)

            orbit_type = get_orbit_type(
                tle_struct["mean_motion"],
                tle_struct["inclination"],
                tle_struct["eccentricity"],
            )
            threshold = self._threshold_for(orbit_type)

            # OOD check before normalization
            ood_info   = self.normalizer.check_ood(vector)
            aug_vector = _make_augmented_vector(vector, orbit_type)
            vec_norm   = self.normalizer.transform(np.array(aug_vector))
            vec_tensor = torch.tensor(vec_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

            # model inference
            with torch.no_grad():
                reconstructed, latent = self.model(vec_tensor)
                recon_error   = torch.mean((vec_tensor - reconstructed) ** 2).item()
                anomaly_score = self.model.anomaly_detector(latent).item()

            reconstructed_denorm = self.normalizer.inverse_transform(
                reconstructed.cpu().numpy()[0]
            )

            # confidence arbitration
            is_anomaly        = recon_error > threshold
            physics_ok        = _physical_status(tle_struct, orbit_type)
            is_maneuver, maneuver_msg = _station_keeping_maneuver(tle_struct, orbit_type)
            ai_low_confidence = is_anomaly and physics_ok and not ood_info["is_ood"]
            ai_ood_override   = ood_info["is_ood"] and ood_info["max_z_score"] > _OOD_Z_WARN

            if ai_ood_override:
                confidence = "Низкое (вход вне области обучения)"
                effective_status = "Требует переобучения модели"
            elif ai_low_confidence:
                confidence = "Низкое (физические параметры в норме)"
                effective_status = "Штатное функционирование"
            elif is_anomaly and anomaly_score >= 0.8:
                confidence = "Высокое"
                effective_status = "Аномалия"
            elif is_anomaly:
                confidence = "Среднее"
                effective_status = "Подозрительно"
            else:
                confidence = "Высокое"
                effective_status = "Штатное функционирование"

            notes = []
            if ood_info["is_ood"]:
                notes.append(
                    f"⚠ Вход вне области обучения (max z={ood_info['max_z_score']:.1f}): "
                    + ", ".join(f[0] for f in ood_info["ood_features"])
                )
            if ai_low_confidence:
                notes.append(
                    "ИИ: Низкое доверие — физические параметры в норме, "
                    "результат модели не подтверждён."
                )
            if is_maneuver:
                notes.append(maneuver_msg)

            if verbose:
                report = generate_detailed_text_with_values(
                    tle_struct,
                    reconstruction_error=recon_error,
                    anomaly_score=anomaly_score,
                    threshold=threshold,
                    orbit_type=orbit_type,
                    confidence=confidence,
                    effective_status=effective_status,
                    notes=notes,
                )
                print(report)
                print()

            return {
                "name":                  name_line.strip(),
                "status":                "analyzed",
                "orbit_type":            orbit_type,
                "orbit_type_human":      classify_orbit(
                                             tle_struct["inclination"],
                                             tle_struct["mean_motion"],
                                             tle_struct["eccentricity"],
                                         ),
                "tle_struct":            tle_struct,
                "vector_original":       vector,
                "vector_reconstructed":  reconstructed_denorm[:6].tolist(),
                "reconstruction_error":  recon_error,
                "anomaly_score":         anomaly_score,
                "is_anomaly":            is_anomaly,
                "threshold_used":        threshold,
                "confidence":            confidence,
                "effective_status":      effective_status,
                "ood_info":              ood_info,
                "notes":                 notes,
                "latent_representation": latent.cpu().numpy()[0].tolist(),
            }

        except TLEParseError as e:
            if verbose:
                print(f"Ошибка парсинга: {name_line}\n  {e}\n")
            return {"name": name_line.strip(), "status": "parse_error", "error": str(e)}
        except Exception as e:
            if verbose:
                print(f"Непредвиденная ошибка: {name_line}\n  {e}\n")
            return {"name": name_line.strip(), "status": "error", "error": str(e)}

    def analyze_file(self, file_path, output_file=None, only_anomalies=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        print(f"АНАЛИЗ ФАЙЛА: {file_path}")
        print(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]

        results = []

        for i in range(0, len(lines) - 2, 3):
            result = self.analyze_single_tle(
                lines[i], lines[i + 1], lines[i + 2],
                verbose=not only_anomalies,
            )
            results.append(result)

            if only_anomalies and result.get("is_anomaly", False):
                r = result
                report = generate_detailed_text_with_values(
                    r["tle_struct"],
                    reconstruction_error=r["reconstruction_error"],
                    anomaly_score=r["anomaly_score"],
                    threshold=r["threshold_used"],
                    orbit_type=r["orbit_type"],
                    confidence=r.get("confidence", "—"),
                    effective_status=r.get("effective_status", "—"),
                    notes=r.get("notes", []),
                )
                print(report)
                print()

        total     = len(results)
        analyzed  = sum(1 for r in results if r["status"] == "analyzed")
        errors    = sum(1 for r in results if r["status"] in ("parse_error", "error"))
        anomalies = sum(1 for r in results if r.get("is_anomaly", False))

        print("СТАТИСТИКА АНАЛИЗА")
        print(f"Всего записей:       {total}")
        print(f"Проанализировано:    {analyzed}")
        print(f"Аномалий:            {anomalies}")
        print(f"Ошибок парсинга:     {errors}")

        from collections import Counter
        orbit_counts = Counter(
            r.get("orbit_type", "?") for r in results if r["status"] == "analyzed"
        )
        if orbit_counts:
            print("По типам орбит:")
            for ot, cnt in sorted(orbit_counts.items()):
                anom = sum(1 for r in results
                           if r.get("orbit_type") == ot and r.get("is_anomaly", False))
                print(f"  {ot:10s}: {cnt:4d}  (аномалий: {anom})")

        if output_file:
            import json
            safe = [
                {k: v for k, v in r.items()
                 if k not in ("tle_struct", "latent_representation")}
                for r in results
            ]
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(safe, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="TLE anomaly analysis")
    parser.add_argument("--file",           "-f", default="data/test.txt")
    parser.add_argument("--model",          "-m", default="model/tle_model.pth")
    parser.add_argument("--normalizer",     "-n", default="model/normalizer.pkl")
    parser.add_argument("--output",         "-o", default=None)
    parser.add_argument("--only-anomalies", "-a", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        system = TLEAnalysisSystem(
            model_path=args.model,
            normalizer_path=args.normalizer,
            device=device,
        )
        system.analyze_file(
            file_path=args.file,
            output_file=args.output,
            only_anomalies=args.only_anomalies,
        )
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}")
        return 1
    except Exception as e:
        print(f"НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())