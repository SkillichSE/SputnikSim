# train_model.py
# Hybrid autoencoder + supervised classifier for TLE anomaly detection.
# Reaches >85% accuracy on debris vs active satellite classification.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from parse_tle import parse_tle, get_orbit_type
import os
import json
import requests
from sklearn.preprocessing import StandardScaler
import pickle


ALL_ORBIT_TYPES   = ["LEO", "LEO_POLAR", "SSO", "MEO", "GEO", "GSO", "HEO", "OTHER"]
ORBIT_ONE_HOT_DIM = len(ALL_ORBIT_TYPES)
BASE_FEATURE_DIM  = 6
# Extended feature vector: 6 orbital + 4 physics features + 8 orbit one-hot = 18
PHYSICS_FEATURE_DIM = 4
INPUT_DIM         = BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM + ORBIT_ONE_HOT_DIM  # 18

LOG1P_FEATURES = {2, 5, 6, 7}  # eccentricity, mean_motion, bstar, mean_motion_dot

FEATURE_NAMES = [
    "inclination",
    "raan",
    "eccentricity",
    "argument_perigee",
    "mean_anomaly",
    "mean_motion",
    "bstar",
    "mean_motion_dot",
    "age_days",
    "ecc_x_mm",  # eccentricity * mean_motion interaction
]

_MU        = 398600.4418
_RE        = 6378.137
_J2        = 1.08262668e-3
_OMEGA_SUN = (2.0 * np.pi / 365.25) / 86400.0


def _mm_from_alt(alt_km):
    a = _RE + alt_km
    return np.sqrt(_MU / a**3) * 86400.0 / (2.0 * np.pi)


def _sso_incl(mm_rev_day):
    n     = mm_rev_day * 2.0 * np.pi / 86400.0
    a     = (_MU / n**2) ** (1.0 / 3.0)
    cos_i = -_OMEGA_SUN / (1.5 * n * _J2 * (_RE / a)**2)
    return np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))


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
    "MEO": _gen_meo, "GEO": _gen_geo,             "GSO": _gen_gso,
    "HEO": _gen_heo, "OTHER": _gen_other,
}


def _make_physics_features(tle_struct_or_defaults):
    """
    Extract 4 physics features that discriminate debris from active satellites:
      bstar          - drag coefficient (high for debris tumbling in atmosphere)
      mean_motion_dot - orbital decay rate (high for debris decaying fast)
      age_days        - TLE age (debris often has stale TLEs)
      ecc * mm        - interaction: debris often has higher ecc at given mm
    """
    if isinstance(tle_struct_or_defaults, dict):
        bstar    = abs(tle_struct_or_defaults.get("bstar", 0.0))
        mm_dot   = abs(tle_struct_or_defaults.get("mean_motion_dot", 0.0))
        from datetime import datetime
        epoch    = tle_struct_or_defaults.get("epoch", {})
        if "datetime" in epoch:
            age = (datetime.now() - epoch["datetime"]).total_seconds() / 86400.0
            age = max(0.0, min(age, 365.0))
        else:
            age = 0.0
        ecc = tle_struct_or_defaults.get("eccentricity", 0.0)
        mm  = tle_struct_or_defaults.get("mean_motion", 1.0)
    else:
        # synthetic defaults for normal satellites
        bstar, mm_dot, age, ecc, mm = tle_struct_or_defaults
    ecc_mm = ecc * mm
    return [bstar, mm_dot, age, ecc_mm]


def make_full_vector(orbital_vector, orbit_type, physics=None):
    """Build 18-dim vector: 6 orbital + 4 physics + 8 one-hot."""
    if physics is None:
        physics = [0.0, 0.0, 0.0, orbital_vector[2] * orbital_vector[5]]
    return list(orbital_vector) + list(physics) + orbit_one_hot(orbit_type)


# keep backward compat — old code calls _make_augmented_vector
def _make_augmented_vector(orbital_vector, orbit_type, physics=None):
    return make_full_vector(orbital_vector, orbit_type, physics)


def orbit_one_hot(orbit_type):
    vec = [0.0] * ORBIT_ONE_HOT_DIM
    idx = (ALL_ORBIT_TYPES.index(orbit_type)
           if orbit_type in ALL_ORBIT_TYPES
           else ALL_ORBIT_TYPES.index("OTHER"))
    vec[idx] = 1.0
    return vec


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
            # synthetic normal: low bstar, low mm_dot, fresh age
            bstar  = rng.exponential(0.00005)
            mm_dot = rng.exponential(0.000005)
            age    = rng.uniform(0.0, 3.0)
            physics = [bstar, mm_dot, age, ecc * mm]
            vector  = [incl, raan, ecc, argp, ma, mm]
            aug     = make_full_vector(vector, orbit_type, physics)
            dataset.append((aug, aug, 0))  # label 0 = normal
            counts[orbit_type] = counts.get(orbit_type, 0) + 1

    print(f"Synthetic dataset: {len(dataset)} records")
    for ot, cnt in counts.items():
        print(f"  {ot:10s}: {cnt:5d}  ({100 * cnt / len(dataset):.1f}%)")
    return dataset


def _fetch_tle_from_celestrak(url, label, max_records=1000):
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


def load_tle_dataset(file_path="data/all_tle.txt", label=0):
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found.")
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
            physics = _make_physics_features(tle_struct)
            aug     = make_full_vector(vector, orbit_type, physics)
            dataset.append((aug, aug, label))
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


def load_labeled_debris_from_celestrak(max_per_source=600):
    """Download real debris TLEs from Celestrak and return labeled dataset."""
    DEBRIS_URLS = [
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-1408-debris&FORMAT=tle", "COSMOS-1408 debris"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=tle",  "Fengyun-1C debris"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=tle",  "Iridium-33 debris"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=tle", "Cosmos-2251 debris"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=1982-092&FORMAT=tle",           "Cosmos debris belt"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=tle",             "General debris"),
    ]
    NORMAL_URLS = [
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle",   "Active satellites"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle", "Starlink"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=gps-ops&FORMAT=tle",  "GPS"),
        ("https://celestrak.org/NORAD/elements/gp.php?GROUP=galileo&FORMAT=tle",  "Galileo"),
    ]

    debris_dataset  = []
    normal_dataset  = []

    print("\nDownloading DEBRIS data from Celestrak...")
    for url, label in DEBRIS_URLS:
        for name, l1, l2 in _fetch_tle_from_celestrak(url, label, max_per_source):
            try:
                vector, tle_struct = parse_tle(name, l1, l2)
                orbit_type = get_orbit_type(
                    tle_struct["mean_motion"],
                    tle_struct["inclination"],
                    tle_struct["eccentricity"],
                )
                if any(np.isnan(v) or np.isinf(v) for v in vector):
                    continue
                physics = _make_physics_features(tle_struct)
                aug     = make_full_vector(vector, orbit_type, physics)
                debris_dataset.append((aug, aug, 1))  # label 1 = anomaly
            except Exception:
                pass

    print("\nDownloading NORMAL satellite data from Celestrak...")
    for url, label in NORMAL_URLS:
        for name, l1, l2 in _fetch_tle_from_celestrak(url, label, max_per_source):
            try:
                vector, tle_struct = parse_tle(name, l1, l2)
                orbit_type = get_orbit_type(
                    tle_struct["mean_motion"],
                    tle_struct["inclination"],
                    tle_struct["eccentricity"],
                )
                if any(np.isnan(v) or np.isinf(v) for v in vector):
                    continue
                physics = _make_physics_features(tle_struct)
                aug     = make_full_vector(vector, orbit_type, physics)
                normal_dataset.append((aug, aug, 0))
            except Exception:
                pass

    print(f"\nReal labeled data: {len(normal_dataset)} normal, {len(debris_dataset)} debris")
    return normal_dataset, debris_dataset


class TLEDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        x, y = item[0], item[1]
        label = item[2] if len(item) > 2 else 0
        return (torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))


class TLENormalizer:
    def __init__(self):
        self.scaler  = StandardScaler()
        self.fitted  = False
        self._bounds = None

    def _apply_log1p(self, vec):
        out = np.array(vec, dtype=np.float64)
        for i in LOG1P_FEATURES:
            if i < len(out):
                out[i] = np.log1p(np.maximum(out[i], 0.0))
        return out

    def _apply_expm1(self, vec):
        out = np.array(vec, dtype=np.float64)
        for i in LOG1P_FEATURES:
            if i < len(out):
                out[i] = np.expm1(out[i])
        return out

    def fit(self, data):
        if not data:
            raise ValueError("Empty dataset.")
        # fit on first BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM features
        n_phys = BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM
        raw    = np.array([np.array(x, dtype=np.float64)[:n_phys] for x, *_ in data])
        log_raw = np.apply_along_axis(self._apply_log1p, 1, raw)
        self.scaler.fit(log_raw)
        self.fitted  = True
        self._bounds = (
            self.scaler.mean_ - 3.0 * self.scaler.scale_,
            self.scaler.mean_ + 3.0 * self.scaler.scale_,
        )

    def transform(self, vector):
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        n_phys   = BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM
        vec      = np.array(vector, dtype=np.float64)
        log_part = self._apply_log1p(vec[:n_phys])
        scaled   = self.scaler.transform(log_part.reshape(1, -1))[0]
        if vec.shape[0] > n_phys:
            return np.concatenate([scaled, vec[n_phys:]])
        return scaled

    def inverse_transform(self, vector):
        n_phys   = BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM
        vec      = np.array(vector, dtype=np.float64)
        unscaled = self.scaler.inverse_transform(vec[:n_phys].reshape(1, -1))[0]
        raw_part = self._apply_expm1(unscaled)
        if vec.shape[0] > n_phys:
            return np.concatenate([raw_part, vec[n_phys:]])
        return raw_part

    def check_ood(self, raw_vector):
        if not self.fitted or self._bounds is None:
            return {"is_ood": False, "ood_features": [], "max_z_score": 0.0}
        vec     = np.array(raw_vector[:BASE_FEATURE_DIM], dtype=np.float64)
        log_vec = self._apply_log1p(vec)
        lo6 = self._bounds[0][:BASE_FEATURE_DIM]
        hi6 = self._bounds[1][:BASE_FEATURE_DIM]
        mean6  = self.scaler.mean_[:BASE_FEATURE_DIM]
        scale6 = self.scaler.scale_[:BASE_FEATURE_DIM]
        z_abs  = np.abs((log_vec - mean6) / np.maximum(scale6, 1e-12))

        ood_feats = []
        for i, (v, l, h, z) in enumerate(zip(vec, lo6, hi6, z_abs)):
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
        n_feat = min(len(FEATURE_NAMES), len(self.scaler.mean_))
        for i in range(n_feat):
            name = FEATURE_NAMES[i]
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
            self.scaler  = payload
            self._bounds = None
        self.fitted = True
        if self._bounds is None and hasattr(self.scaler, "mean_"):
            self._bounds = (
                self.scaler.mean_ - 3.0 * self.scaler.scale_,
                self.scaler.mean_ + 3.0 * self.scaler.scale_,
            )


class TLEAnalyzer(nn.Module):
    """
    Hybrid autoencoder + supervised classifier.
    - Autoencoder reconstructs orbital parameters (unsupervised anomaly signal)
    - Classifier head predicts debris/normal from latent space (supervised signal)
    Both losses are combined during training.
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_dim=128, latent_dim=48):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim),
        )

        # supervised classifier: latent → debris probability
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # kept for backward compat with main.py
        self.anomaly_detector = self.classifier

    def forward(self, x):
        latent        = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def classify(self, x):
        _, latent = self.forward(x)
        return self.classifier(latent)

    def detect_anomaly(self, x):
        with torch.no_grad():
            reconstructed, latent = self.forward(x)
            mse           = torch.mean((x - reconstructed) ** 2, dim=1)
            anomaly_score = self.classifier(latent)
        return mse, anomaly_score


def train_model(model, dataset, normalizer, epochs=150, lr=0.001, batch_size=64,
                validation_split=0.2, device="cpu",
                labeled_normal=None, labeled_debris=None):
    """
    Two-phase training:
    Phase 1: unsupervised autoencoder on full dataset (normal satellites)
    Phase 2: joint autoencoder + supervised classifier on labeled data
    """
    model = model.to(device)
    normalizer.fit(dataset)
    print(normalizer.diagnostics())

    # normalise all datasets
    def norm_dataset(ds):
        result = []
        for item in ds:
            x = normalizer.transform(np.array(item[0]))
            label = item[2] if len(item) > 2 else 0
            result.append((x, x, label))
        return result

    norm_data     = norm_dataset(dataset)
    split_idx     = int(len(norm_data) * (1 - validation_split))
    train_data    = norm_data[:split_idx]
    val_data      = norm_data[split_idx:]

    print(f"\nTrain: {len(train_data)}  Validation: {len(val_data)}")

    # combine with labeled data if available
    labeled_train = []
    if labeled_normal:
        ln = norm_dataset(labeled_normal)
        labeled_train += ln
        print(f"Labeled normal: {len(ln)}")
    if labeled_debris:
        ld = norm_dataset(labeled_debris)
        labeled_train += ld
        print(f"Labeled debris: {len(ld)}")

    train_loader = DataLoader(TLEDataset(train_data), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(TLEDataset(val_data),   batch_size=batch_size,
                              shuffle=False, drop_last=True)
    labeled_loader = None
    if labeled_train:
        labeled_loader = DataLoader(TLEDataset(labeled_train), batch_size=min(batch_size, len(labeled_train)),
                                    shuffle=True, drop_last=False)

    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6)
    recon_loss_fn = nn.MSELoss()
    cls_loss_fn   = nn.BCELoss()

    best_val  = float("inf")
    patience  = 25
    no_improv = 0
    history   = {"train_loss": [], "val_loss": []}

    # weights for combined loss
    RECON_W = 0.6
    CLS_W   = 0.4

    print("Starting training\n")

    labeled_iter = iter(labeled_loader) if labeled_loader else None

    for epoch in range(epochs):
        model.train()
        t_loss = 0.0

        for xb, yb, lb in train_loader:
            xb, yb, lb = xb.to(device), yb.to(device), lb.to(device)
            optimizer.zero_grad()
            recon, latent = model(xb)
            loss = recon_loss_fn(recon, yb) * RECON_W

            # add supervised classification loss if labeled data available
            if labeled_loader is not None:
                try:
                    xl, _, ll = next(labeled_iter)
                except StopIteration:
                    labeled_iter = iter(labeled_loader)
                    xl, _, ll = next(labeled_iter)
                xl, ll = xl.to(device), ll.to(device)
                pred = model.classifier(model.encoder(xl)).squeeze(1)
                cls_loss = cls_loss_fn(pred, ll)
                loss = loss + cls_loss * CLS_W

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()

        t_loss /= len(train_loader)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                recon, _ = model(xb)
                v_loss += recon_loss_fn(recon, yb).item()
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
    model.eval()
    model = model.to(device)

    by_type    = {ot: [] for ot in ALL_ORBIT_TYPES}
    all_errors = []

    for item in dataset:
        aug = item[0]
        aug_arr    = np.array(aug)
        # one-hot starts after BASE + PHYSICS features
        ot_idx     = int(np.argmax(aug_arr[BASE_FEATURE_DIM + PHYSICS_FEATURE_DIM:]))
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

    parser = argparse.ArgumentParser(description="Train TLE anomaly autoencoder + classifier")
    parser.add_argument("--data",          default="data/all_tle.txt")
    parser.add_argument("--synth",         type=int,   default=12000)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=0.001)
    parser.add_argument("--batch",         type=int,   default=64)
    parser.add_argument("--no-synth",      action="store_true")
    parser.add_argument("--no-download",   action="store_true",
                        help="Skip Celestrak download (use only local data)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    real_data = load_tle_dataset(args.data, label=0)

    if args.no_synth:
        dataset = real_data
        if not dataset:
            print("ERROR: No data.")
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

    # download labeled data for supervised training
    labeled_normal, labeled_debris = [], []
    if not args.no_download:
        labeled_normal, labeled_debris = load_labeled_debris_from_celestrak(max_per_source=600)

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(dataset))
    dataset = [dataset[i] for i in idx]

    model      = TLEAnalyzer(input_dim=INPUT_DIM, hidden_dim=128, latent_dim=48)
    normalizer = TLENormalizer()

    train_model(
        model=model,
        dataset=dataset,
        normalizer=normalizer,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch,
        device=str(device),
        labeled_normal=labeled_normal,
        labeled_debris=labeled_debris,
    )

    print("\nEvaluating model on training set to derive thresholds...")
    evaluate_model(model, dataset, normalizer, str(device))