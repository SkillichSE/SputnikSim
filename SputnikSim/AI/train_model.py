# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from parse_tle import parse_tle, get_orbit_type
import os
import json
from sklearn.preprocessing import StandardScaler
import pickle


ALL_ORBIT_TYPES   = ["LEO", "LEO_POLAR", "SSO", "MEO", "GEO", "GSO", "HEO", "OTHER"]
ORBIT_ONE_HOT_DIM = len(ALL_ORBIT_TYPES)
BASE_FEATURE_DIM  = 6
INPUT_DIM         = BASE_FEATURE_DIM + ORBIT_ONE_HOT_DIM  # 14

LOG1P_FEATURES = {2, 5}  # eccentricity idx 2  mean_motion idx 5

FEATURE_NAMES = [
    "inclination",       # deg
    "raan",              # deg
    "eccentricity",      # log1p
    "argument_perigee",  # deg
    "mean_anomaly",      # deg
    "mean_motion",       # rev/d  log1p
]

_MU        = 398600.4418
_RE        = 6378.137
_J2        = 1.08262668e-3
_OMEGA_SUN = (2.0 * np.pi / 365.25) / 86400.0  # rad/s


def _mm_from_alt(alt_km):
    """rev/day for circular orbit at alt_km"""
    a = _RE + alt_km
    return np.sqrt(_MU / a**3) * 86400.0 / (2.0 * np.pi)


def _sso_incl(mm_rev_day):
    """theoretical SSO inclination from J2"""
    n     = mm_rev_day * 2.0 * np.pi / 86400.0
    a     = (_MU / n**2) ** (1.0 / 3.0)
    cos_i = -_OMEGA_SUN / (1.5 * n * _J2 * (_RE / a)**2)
    return np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))


# fractions from Celestrak catalog
_ORBIT_POPULATION = {
    "LEO":       0.38,
    "LEO_POLAR": 0.12,
    "SSO":       0.18,
    "MEO":       0.07,
    "GEO":       0.12,
    "GSO":       0.06,
    "HEO":       0.04,
    "OTHER":     0.03,
}


def _gen_leo(n, rng):
    samples = []
    for _ in range(n):
        alt  = rng.uniform(200, 1800)
        mm   = _mm_from_alt(alt) * rng.uniform(0.9998, 1.0002)
        incl = rng.uniform(0.0, 65.0)
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.002), 0.09)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_leo_polar(n, rng):
    """incl 80-98 excluding SSO band"""
    samples = []
    for _ in range(n):
        alt  = rng.uniform(200, 1600)
        mm   = _mm_from_alt(alt) * rng.uniform(0.9998, 1.0002)
        while True:
            incl = rng.uniform(80.0, 98.0)
            if not (97.5 <= incl <= 99.5):
                break
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.002), 0.09)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_sso(n, rng):
    """incl coupled to altitude via J2"""
    samples = []
    for _ in range(n):
        alt  = rng.uniform(300, 1000)
        mm   = _mm_from_alt(alt) * rng.uniform(0.9998, 1.0002)
        incl = np.clip(_sso_incl(mm) + rng.normal(0.0, 0.15), 97.0, 100.5)
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.0005), 0.02)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_meo(n, rng):
    """GPS/GLONASS/Galileo/BeiDou"""
    samples = []
    constellations = [
        (19100, 20300, 55.0, 0.5),
        (19100, 19200, 64.8, 0.5),
        (23222, 23222, 56.0, 0.3),
        (21528, 21528, 55.5, 0.3),
    ]
    for i in range(n):
        alt_lo, alt_hi, incl_nom, incl_sig = constellations[i % len(constellations)]
        alt  = rng.uniform(alt_lo, alt_hi)
        mm   = _mm_from_alt(alt) * rng.uniform(0.9998, 1.0002)
        incl = np.clip(rng.normal(incl_nom, incl_sig), 50.0, 70.0)
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.001), 0.02)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_geo(n, rng):
    samples = []
    for _ in range(n):
        mm   = rng.normal(1.0027, 0.0015)
        incl = min(np.abs(rng.normal(0.0, 0.3)), 4.5)
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.0003), 0.005)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_gso(n, rng):
    """graveyard / IGSO"""
    samples = []
    for _ in range(n):
        mm   = rng.normal(1.0027, 0.08)
        incl = rng.uniform(0.0, 30.0)
        raan = rng.uniform(0.0, 360.0)
        ecc  = min(rng.exponential(0.005), 0.08)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_heo(n, rng):
    """Molniya 60%  Tundra 40%"""
    samples = []
    for _ in range(n):
        is_molniya = rng.random() < 0.6
        if is_molniya:
            mm   = rng.normal(2.0, 0.05)
            ecc  = rng.normal(0.72, 0.02)
            incl = rng.normal(63.4, 1.0)
        else:
            mm   = rng.normal(1.002, 0.03)
            ecc  = rng.uniform(0.25, 0.45)
            incl = rng.uniform(28.0, 64.0)
        ecc  = np.clip(ecc, 0.25, 0.85)
        incl = np.clip(incl, 0.0, 90.0)
        raan = rng.uniform(0.0, 360.0)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


def _gen_other(n, rng):
    """deep space / libration"""
    samples = []
    for _ in range(n):
        mm   = rng.uniform(0.1, 0.85)
        incl = rng.uniform(0.0, 90.0)
        raan = rng.uniform(0.0, 360.0)
        ecc  = rng.uniform(0.0, 0.9)
        argp = rng.uniform(0.0, 360.0)
        ma   = rng.uniform(0.0, 360.0)
        samples.append([incl, raan, ecc, argp, ma, mm])
    return samples


_GEN_FUNCS = {
    "LEO": _gen_leo, "LEO_POLAR": _gen_leo_polar, "SSO": _gen_sso,
    "MEO": _gen_meo, "GEO": _gen_geo,       "GSO": _gen_gso,
    "HEO": _gen_heo, "OTHER": _gen_other,
}


def generate_synthetic_dataset(n_total=12000, seed=42):
    rng     = np.random.default_rng(seed)
    dataset = []
    counts  = {}

    for orbit_type, fraction in _ORBIT_POPULATION.items():
        n           = max(1, int(n_total * fraction))
        raw_samples = _GEN_FUNCS[orbit_type](n, rng)
        for raw in raw_samples:
            incl, raan, ecc, argp, ma, mm = raw
            if not (0.0 <= incl <= 180.0): continue
            if not (0.0 <= ecc < 1.0):     continue
            if not (mm > 0):               continue
            vector = [incl, raan, ecc, argp, ma, mm]
            aug    = _make_augmented_vector(vector, orbit_type)
            dataset.append((aug, aug))
            counts[orbit_type] = counts.get(orbit_type, 0) + 1

    print(f"Synthetic dataset: {len(dataset)} records")
    for ot, cnt in counts.items():
        print(f"  {ot:10s}: {cnt:5d}  ({100 * cnt / len(dataset):.1f}%)")

    return dataset


def orbit_one_hot(orbit_type):
    vec = [0.0] * ORBIT_ONE_HOT_DIM
    idx = (ALL_ORBIT_TYPES.index(orbit_type)
           if orbit_type in ALL_ORBIT_TYPES
           else ALL_ORBIT_TYPES.index("OTHER"))
    vec[idx] = 1.0
    return vec


def _make_augmented_vector(orbital_vector, orbit_type):
    """6-dim params + 8-dim one-hot"""
    return list(orbital_vector) + orbit_one_hot(orbit_type)


def load_tle_dataset(file_path="data/all_tle.txt"):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found — using synthetic data only.")
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]

    print(f"Read {len(lines)} lines from {file_path}")

    dataset = []
    errors  = []
    counts  = {k: 0 for k in ALL_ORBIT_TYPES}

    for i in range(0, len(lines) - 2, 3):
        try:
            vector, tle_struct = parse_tle(lines[i], lines[i+1], lines[i+2])
            orbit_type = get_orbit_type(
                tle_struct["mean_motion"],
                tle_struct["inclination"],
                tle_struct["eccentricity"],
            )
            if any(np.isnan(v) or np.isinf(v) for v in vector):
                continue
            aug = _make_augmented_vector(vector, orbit_type)
            dataset.append((aug, aug))
            counts[orbit_type] = counts.get(orbit_type, 0) + 1
        except Exception as e:
            errors.append(str(e))

    print(f"Real TLE records: {len(dataset)}")
    for ot, cnt in counts.items():
        if cnt:
            print(f"  {ot:10s}: {cnt}")
    if errors:
        print(f"  Parse errors: {len(errors)}")
    return dataset


class TLEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))


class TLENormalizer:
    def __init__(self):
        self.scaler  = StandardScaler()
        self.fitted  = False
        self._bounds = None  # 3-sigma in log1p space for OOD

    def _apply_log1p(self, vec6):
        out = vec6.copy().astype(np.float64)
        for i in LOG1P_FEATURES:
            out[i] = np.log1p(np.maximum(out[i], 0.0))
        return out

    def _apply_expm1(self, vec6):
        out = vec6.copy().astype(np.float64)
        for i in LOG1P_FEATURES:
            out[i] = np.expm1(out[i])
        return out

    def fit(self, data):
        if not data:
            raise ValueError("Empty dataset — cannot fit normalizer.")

        raw     = np.array([np.array(x, dtype=np.float64)[:BASE_FEATURE_DIM] for x, _ in data])
        log_raw = np.apply_along_axis(self._apply_log1p, 1, raw)
        self.scaler.fit(log_raw)
        self.fitted  = True
        self._bounds = (
            self.scaler.mean_ - 3.0 * self.scaler.scale_,
            self.scaler.mean_ + 3.0 * self.scaler.scale_,
        )

    def transform(self, vector):
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        vec      = np.array(vector, dtype=np.float64)
        log_part = self._apply_log1p(vec[:BASE_FEATURE_DIM])
        scaled   = self.scaler.transform(log_part.reshape(1, -1))[0]

        if vec.shape[0] > BASE_FEATURE_DIM:
            return np.concatenate([scaled, vec[BASE_FEATURE_DIM:]])
        return scaled

    def inverse_transform(self, vector):
        vec      = np.array(vector, dtype=np.float64)
        unscaled = self.scaler.inverse_transform(vec[:BASE_FEATURE_DIM].reshape(1, -1))[0]
        raw_part = self._apply_expm1(unscaled)

        if vec.shape[0] > BASE_FEATURE_DIM:
            return np.concatenate([raw_part, vec[BASE_FEATURE_DIM:]])
        return raw_part

    def check_ood(self, raw_vector):
        if not self.fitted or self._bounds is None:
            return {"is_ood": False, "ood_features": [], "max_z_score": 0.0}

        vec     = np.array(raw_vector[:BASE_FEATURE_DIM], dtype=np.float64)
        log_vec = self._apply_log1p(vec)
        lo, hi  = self._bounds
        z_abs   = np.abs((log_vec - self.scaler.mean_) /
                         np.maximum(self.scaler.scale_, 1e-12))

        ood_feats = []
        for i, (v, l, h, z) in enumerate(zip(vec, lo, hi, z_abs)):
            lv = log_vec[i]
            if lv < l or lv > h:
                name = FEATURE_NAMES[i] if i < len(FEATURE_NAMES) else f"feat_{i}"
                ood_feats.append((name, float(v), float(np.expm1(l)), float(np.expm1(h))))

        return {
            "is_ood":       len(ood_feats) > 0,
            "ood_features": ood_feats,
            "max_z_score":  float(np.max(z_abs)),
        }

    def diagnostics(self):
        if not self.fitted:
            return "Normalizer: NOT FITTED"
        lines = ["Normalizer statistics (training distribution, log1p space):"]
        for i, name in enumerate(FEATURE_NAMES):
            m  = self.scaler.mean_[i]
            s  = self.scaler.scale_[i]
            lo = self._bounds[0][i] if self._bounds else m - 3*s
            hi = self._bounds[1][i] if self._bounds else m + 3*s
            if i in LOG1P_FEATURES:
                lo_raw, hi_raw, tag = np.expm1(lo), np.expm1(hi), " [log1p]"
            else:
                lo_raw, hi_raw, tag = lo, hi, ""
            lines.append(
                f"  {name:20s}{tag}: mean={m:8.4f}  std={s:7.4f}  "
                f"3σ raw=[{lo_raw:.3f}, {hi_raw:.3f}]"
            )
        return "\n".join(lines)

    def save(self, path):
        payload = {"scaler": self.scaler, "bounds": self._bounds,
                   "log1p_features": list(LOG1P_FEATURES)}
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "scaler" in payload:
            self.scaler  = payload["scaler"]
            self._bounds = payload.get("bounds")
        else:
            self.scaler  = payload  # legacy bare scaler
            self._bounds = None
        self.fitted = True
        if self._bounds is None and hasattr(self.scaler, "mean_"):
            self._bounds = (
                self.scaler.mean_ - 3.0 * self.scaler.scale_,
                self.scaler.mean_ + 3.0 * self.scaler.scale_,
            )


class TLEAnalyzer(nn.Module):
    """autoencoder for anomaly detection"""

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, latent_dim=48):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, input_dim),
        )

        self.anomaly_detector = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent        = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def detect_anomaly(self, x):
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            mse           = torch.mean((x - reconstructed) ** 2, dim=1)
            anomaly_score = self.anomaly_detector(latent)
        return mse, anomaly_score


def train_model(model, dataset, normalizer, epochs=150, lr=0.001, batch_size=64,
                validation_split=0.2, device="cpu"):

    model = model.to(device)
    normalizer.fit(dataset)
    print(normalizer.diagnostics())

    normalised = [(normalizer.transform(np.array(x)), normalizer.transform(np.array(x)))
                  for x, _ in dataset]

    split_idx  = int(len(normalised) * (1 - validation_split))
    train_data = normalised[:split_idx]
    val_data   = normalised[split_idx:]

    print(f"\nTrain: {len(train_data)}  Validation: {len(val_data)}")

    train_loader = DataLoader(TLEDataset(train_data), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(TLEDataset(val_data),   batch_size=batch_size,
                              shuffle=False, drop_last=True)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    criterion = nn.MSELoss()

    best_val  = float("inf")
    patience  = 20
    no_improv = 0
    history   = {"train_loss": [], "val_loss": []}

    print("Starting training\n")

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            recon, _ = model(xb)
            loss = criterion(recon, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, _ = model(xb)
                v_loss += criterion(recon, yb).item()
        v_loss /= len(val_loader)

        scheduler.step(v_loss)
        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {t_loss:.6f} | Val: {v_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if v_loss < best_val:
            best_val  = v_loss
            no_improv = 0
            os.makedirs("model", exist_ok=True)
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss":         v_loss,
                "input_dim":        INPUT_DIM,
                "feature_names":    FEATURE_NAMES,
                "log1p_features":   list(LOG1P_FEATURES),
            }, "model/tle_model_best.pth")
        else:
            no_improv += 1
            if no_improv >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print(f"\nTraining complete. Best val loss: {best_val:.6f}")
    torch.save(model.state_dict(), "model/tle_model.pth")
    normalizer.save("model/normalizer.pkl")
    with open("model/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print("Saved: model/tle_model_best.pth, model/tle_model.pth, "
          "model/normalizer.pkl, model/training_history.json")
    return history


def evaluate_model(model, dataset, normalizer, device="cpu"):
    """reconstruction error stats per orbit type"""
    model.eval()
    model = model.to(device)

    by_type    = {ot: [] for ot in ALL_ORBIT_TYPES}
    all_errors = []

    for aug, _ in dataset:
        aug_arr    = np.array(aug)
        ot_idx     = int(np.argmax(aug_arr[BASE_FEATURE_DIM:]))
        orbit_type = ALL_ORBIT_TYPES[ot_idx]

        vec_norm = normalizer.transform(aug_arr)
        t        = torch.tensor(vec_norm, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            recon, _ = model(t)
            err = torch.mean((t - recon) ** 2).item()
        by_type[orbit_type].append(err)
        all_errors.append(err)

    all_errors = np.array(all_errors)
    print("\nEvaluation results:")
    print(f"  ALL  n={len(all_errors):5d}  "
          f"mean={np.mean(all_errors):.5f}  p95={np.percentile(all_errors, 95):.5f}")

    thresholds = {}
    for ot in ALL_ORBIT_TYPES:
        errs = by_type[ot]
        if not errs:
            continue
        errs = np.array(errs)
        p95  = float(np.percentile(errs, 95))
        thresholds[ot] = round(p95, 5)
        print(f"  {ot:10s}  n={len(errs):5d}  "
              f"mean={np.mean(errs):.5f}  p95={p95:.5f}  <- suggested threshold")

    print("\nSuggested ORBIT_THRESHOLDS dict for main.py:")
    print("ORBIT_THRESHOLDS = {")
    for ot in ALL_ORBIT_TYPES:
        print(f'    "{ot}": {thresholds.get(ot, 0.10)},')
    print("}")
    return all_errors


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train TLE anomaly autoencoder")
    parser.add_argument("--data",     default="data/all_tle.txt")
    parser.add_argument("--synth",    type=int, default=12000)
    parser.add_argument("--epochs",   type=int, default=150)
    parser.add_argument("--lr",       type=float, default=0.001)
    parser.add_argument("--batch",    type=int, default=64)
    parser.add_argument("--no-synth", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    real_data = load_tle_dataset(args.data)

    if args.no_synth:
        dataset = real_data
        if not dataset:
            print("ERROR: No data available and --no-synth specified.")
            exit(1)
    else:
        print(f"\nGenerating {args.synth} synthetic TLE records...")
        synth_data = generate_synthetic_dataset(n_total=args.synth)
        dataset    = synth_data + real_data
        print(f"\nTotal dataset: {len(dataset)} records "
              f"({len(synth_data)} synthetic + {len(real_data)} real)\n")

    if not dataset:
        print("ERROR: Empty dataset.")
        exit(1)

    rng     = np.random.default_rng(0)
    idx     = rng.permutation(len(dataset))
    dataset = [dataset[i] for i in idx]

    model      = TLEAnalyzer(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=48)
    normalizer = TLENormalizer()

    train_model(model=model, dataset=dataset, normalizer=normalizer,
                epochs=args.epochs, lr=args.lr, batch_size=args.batch, device=str(device))

    print("\nEvaluating model on training set to derive thresholds...")
    evaluate_model(model, dataset, normalizer, str(device))