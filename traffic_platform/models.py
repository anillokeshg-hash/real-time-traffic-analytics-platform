"""ML models: congestion classifier, incident detector, traffic forecaster."""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from simulator import FREE_SPEED, JUNCTIONS, make_reading

SENSOR_FEATURES = ["count", "speed", "occupancy"]
LEVELS = ["Low", "Medium", "High"]


def congestion_index(speed, occupancy):
    """Ground-truth congestion score from speed and occupancy (0 = free, 1 = jammed)."""
    return 0.6 * (1 - speed / FREE_SPEED) + 0.4 * occupancy / 100


def level_from_index(idx):
    return 0 if idx < 0.35 else 1 if idx < 0.55 else 2


def build_dataset(n=6000, seed=3, incidents=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        j = JUNCTIONS[int(rng.integers(len(JUNCTIONS)))]
        hour = float(rng.uniform(0, 24))
        r = make_reading(j["capacity"], hour, incident=bool(rng.random() < incidents))
        r["hour"] = hour
        r["capacity"] = j["capacity"]
        r["level"] = level_from_index(congestion_index(r["speed"], r["occupancy"]) + rng.normal(0, 0.02))
        rows.append(r)
    return pd.DataFrame(rows)


def train_models(verbose=True):
    # 1) Congestion level classifier
    df = build_dataset()
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[SENSOR_FEATURES], df["level"], test_size=0.2, random_state=3, stratify=df["level"]
    )
    classifier = RandomForestClassifier(n_estimators=100, random_state=3).fit(X_tr, y_tr)

    # 2) Incident detector: IsolationForest learns what normal readings look like
    normal = build_dataset(n=6000, seed=4, incidents=0.0)
    detector = IsolationForest(n_estimators=150, contamination=0.01, random_state=4)
    detector.fit(normal[SENSOR_FEATURES])

    # 3) Forecaster: expected vehicle count from time of day and road capacity
    fdf = build_dataset(n=8000, seed=5)
    F = ["hour", "capacity"]
    f_tr, f_te, t_tr, t_te = train_test_split(fdf[F], fdf["count"], test_size=0.2, random_state=5)
    forecaster = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=5).fit(f_tr, t_tr)

    if verbose:
        print(f"Congestion classifier accuracy: {accuracy_score(y_te, classifier.predict(X_te)):.3f}")
        print(f"Forecast model R2: {r2_score(t_te, forecaster.predict(f_te)):.3f}")
        test_inc = build_dataset(n=1000, seed=6, incidents=1.0)
        test_ok = build_dataset(n=1000, seed=7, incidents=0.0)
        hit = (detector.predict(test_inc[SENSOR_FEATURES]) == -1).mean()
        false_alarm = (detector.predict(test_ok[SENSOR_FEATURES]) == -1).mean()
        print(f"Incident detector: catches {hit:.0%} of incidents, {false_alarm:.1%} false alarms")
    return classifier, detector, forecaster


def classify(classifier, r):
    X = pd.DataFrame([[r[f] for f in SENSOR_FEATURES]], columns=SENSOR_FEATURES)
    return LEVELS[int(classifier.predict(X)[0])]


def is_incident(detector, r):
    X = pd.DataFrame([[r[f] for f in SENSOR_FEATURES]], columns=SENSOR_FEATURES)
    return bool(detector.predict(X)[0] == -1)


def forecast_count(forecaster, hour, capacity):
    X = pd.DataFrame([[hour % 24, capacity]], columns=["hour", "capacity"])
    return max(round(float(forecaster.predict(X)[0])), 0)


if __name__ == "__main__":
    train_models()
