import pandas as pd
from pathlib import Path
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATASET_PATH = Path(__file__).resolve().parents[1] / "dataset" / "house_price_data_100k.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"

TARGET = "Price"
NUM_COLS = ["Area", "Bedrooms", "Bathrooms", "Year Built"]
CAT_COLS = ["Location", "Nearby Metro"]

def load_data():
    df = pd.read_csv(DATASET_PATH)
    return df

def build_pipeline():
    numeric = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(transformers=[
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS)
    ])
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    pipe = Pipeline([("preproc", pre), ("model", model)])
    return pipe

def train_and_save():
    df = load_data()
    X = df[NUM_COLS + CAT_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "r2": float(r2_score(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred))
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    return metrics

if __name__ == "__main__":
    metrics = train_and_save()
    print({"status": "ok", "metrics": metrics, "model_path": str(MODEL_PATH)})
