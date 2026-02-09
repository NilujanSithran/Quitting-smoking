"""
Kaggle Pipeline – single app run by browser.
Serves API and React frontend. Run: python app.py then open http://localhost:8000
"""

import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
app = FastAPI(title="Kaggle Pipeline")


@app.on_event("startup")
def _startup_train():
    """Pre-train prediction models at server start so /api/predict is instant (no delay on first click)."""
    train_path = ROOT / "train.csv"
    if train_path.exists():
        try:
            _train_prediction_models()
        except Exception:
            pass  # train on first /predict if startup fails (e.g. missing columns)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project root (this file's directory); resolve so paths work from any cwd
ROOT = Path(__file__).resolve().parent
DIST = ROOT / "frontend" / "dist"
PLOTS_DIR = (ROOT / "plots").resolve()

# API as a separate app mounted at /api – main app never sees /api/*, so no 405 on POST
api_app = FastAPI(title="Pipeline API")

# Real-time prediction: models pre-trained at server startup so every prediction is instant (no delay in viva/demo).
MAX_TRAIN_ROWS = 1200  # small sample so startup finishes in ~1–2 seconds
PREDICT_FEATURES = [
    "age",
    "waist(cm)",
    "fasting blood sugar",
    "triglyceride",
    "HDL",
    "hemoglobin",
    "Gtp",
]
_prediction_models = None  # { "scaler", "rf", "svm", "nn", "best_model", "best_accuracy" } or None


def _data_path() -> Path:
    """Training uses train.csv only (test.csv has no 'smoking' label)."""
    return ROOT / "train.csv"


def list_plot_files() -> list[str]:
    """Return sorted list of all .png filenames in plots/ so frontend can display them dynamically."""
    if not PLOTS_DIR.is_dir():
        return []
    return sorted(
        f.name for f in PLOTS_DIR.iterdir()
        if f.is_file() and f.suffix.lower() == ".png"
    )


@api_app.get("/health")
def health():
    return {"status": "ok"}


@api_app.get("/stats")
def get_stats():
    """Dataset stats from train.csv."""
    data_path = _data_path()
    if not data_path.exists():
        return {"error": "train.csv not found", "rows": 0, "columns": 0, "target_distribution": {}}
    import pandas as pd
    df = pd.read_csv(data_path, nrows=0)
    row_count = sum(1 for _ in open(data_path)) - 1
    cols = list(df.columns)
    target_col = "smoking"
    target_dist = {}
    if target_col in cols:
        df_full = pd.read_csv(data_path)
        target_dist = df_full[target_col].value_counts().astype(int).to_dict()
    return {
        "rows": int(row_count),
        "columns": len(cols),
        "column_names": cols,
        "target_distribution": target_dist,
    }


@api_app.get("/results")
def get_results():
    """Model comparison results (written by pipeline). Includes plot_files that exist in plots/."""
    results_path = ROOT / "results.json"
    if not results_path.exists():
        data = {
            "available": False,
            "message": "Run the pipeline (pipeline.py) to generate results.",
            "models": [],
            "best_model": None,
        }
    else:
        with open(results_path, "r") as f:
            data = json.load(f)
        data["available"] = bool(data.get("models"))

    # Plot files: dynamically list all .png files in plots/ so all generated graphs are shown (and new ones auto-appear)
    data["plot_files"] = list_plot_files()
    return data


def _train_prediction_models():
    """Train RF, SVM, and NN on a sample of train.csv. Cached. Model accuracies from results.json."""
    global _prediction_models
    if _prediction_models is not None:
        return _prediction_models
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    data_path = _data_path()
    if not data_path.exists():
        raise HTTPException(status_code=503, detail="train.csv not found")
    df = pd.read_csv(data_path, nrows=MAX_TRAIN_ROWS)
    df.columns = df.columns.str.strip()
    target = "smoking"
    missing = [c for c in PREDICT_FEATURES + [target] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=503, detail=f"train.csv missing columns: {missing}")
    df = df[PREDICT_FEATURES + [target]].dropna()
    X = df[PREDICT_FEATURES].astype(np.float32)
    y = df[target].values.astype(np.int32)
    if len(X) < 100:
        raise HTTPException(status_code=503, detail="Not enough rows in train.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight="balanced")
    rf.fit(X_scaled, y)
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42, probability=True)
    svm.fit(X_scaled, y)
    nn_model = None
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        tf.random.set_seed(42)
        nn = keras.Sequential([
            layers.Input(shape=(len(PREDICT_FEATURES),)),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.15),
            layers.Dense(1, activation="sigmoid"),
        ])
        nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        nn.fit(X_scaled, y, epochs=5, batch_size=128, verbose=0)
        nn_model = nn
    except Exception:
        pass
    best_model = "Random Forest"
    best_accuracy = 0.0
    model_accuracies = {}
    results_path = ROOT / "results.json"
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                res = json.load(f)
            best_model = res.get("best_model") or best_model
            best_accuracy = float(res.get("best_accuracy", 0) or 0)
            for m in res.get("models") or []:
                name = m.get("name")
                acc = m.get("accuracy")
                if name and acc is not None:
                    model_accuracies[name] = round(float(acc) * 100, 2)
        except Exception:
            pass
    _prediction_models = {
        "scaler": scaler, "rf": rf, "svm": svm, "nn": nn_model,
        "best_model": best_model, "best_accuracy": best_accuracy, "model_accuracies": model_accuracies,
    }
    return _prediction_models


@api_app.post("/predict")
def predict(data: dict = Body(...)):
    """
    Accept 7 features. Model trained on a sample of train.csv for speed. Returns Smoking/Non-smoking and accuracy.
    """
    import numpy as np

    for key in PREDICT_FEATURES:
        if key not in data:
            raise HTTPException(status_code=400, detail=f"Missing field: {key}")
    try:
        row = np.array([[float(data[k]) for k in PREDICT_FEATURES]], dtype=np.float32)
    except (TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid numbers: {e}")
    models_dict = _train_prediction_models()
    scaler = models_dict["scaler"]
    X = scaler.transform(row)
    predictions = []
    probs_rf = models_dict["rf"].predict_proba(X)[0, 1]
    predictions.append({"model": "Random Forest", "probability_smoking": round(float(probs_rf), 4), "percent": round(probs_rf * 100, 2)})
    probs_svm = models_dict["svm"].predict_proba(X)[0, 1]
    predictions.append({"model": "SVM", "probability_smoking": round(float(probs_svm), 4), "percent": round(probs_svm * 100, 2)})
    probs_nn = None
    if models_dict["nn"] is not None:
        p = models_dict["nn"].predict(X, verbose=0)
        probs_nn = float(p.flatten()[0])
        predictions.append({"model": "Neural Network", "probability_smoking": round(probs_nn, 4), "percent": round(probs_nn * 100, 2)})
    best_model = models_dict.get("best_model") or "Random Forest"
    best_accuracy = models_dict.get("best_accuracy") or 0.0
    if best_model == "Random Forest":
        prob = probs_rf
    elif best_model == "SVM":
        prob = probs_svm
    elif best_model == "Neural Network" and probs_nn is not None:
        prob = probs_nn
    else:
        prob = probs_rf
    predicted_class = 1 if prob >= 0.5 else 0
    model_accuracies = models_dict.get("model_accuracies") or {}
    return {
        "predictions": predictions,
        "predicted_class": predicted_class,
        "predicted_label": "Smoking" if predicted_class == 1 else "Non-smoking",
        "average_probability_smoking": round(float(prob), 4),
        "average_percent": round(prob * 100, 2),
        "model_used": best_model,
        "model_accuracy_percent": round(best_accuracy * 100, 2),
        "model_accuracies": model_accuracies,
    }


@api_app.get("/submission")
def get_submission(limit=100):
    """Preview submission.csv."""
    sub_path = ROOT / "submission.csv"
    if not sub_path.exists():
        return {"available": False, "rows": [], "total": 0}
    import pandas as pd
    df = pd.read_csv(sub_path, nrows=int(limit))
    return {
        "available": True,
        "total": sum(1 for _ in open(sub_path)) - 1,
        "rows": df.to_dict(orient="records"),
    }


# Ensure plots dir exists
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


@api_app.get("/plots/{filename:path}")
def get_plot(filename: str):
    """Serve a single plot image from plots/ so all generated graphs load reliably."""
    if not filename or not filename.endswith(".png"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not a plot image")
    # Restrict to filename only (no path traversal)
    if "/" in filename or "\\" in filename or filename.startswith("."):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Invalid filename")
    path = (PLOTS_DIR / filename).resolve()
    try:
        path.relative_to(PLOTS_DIR.resolve())
    except ValueError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plot not found")
    if not path.is_file():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(
        str(path),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )


app.mount("/api", api_app)


# Serve built frontend so app runs in browser
if DIST.exists():
    assets = DIST / "assets"
    if assets.exists():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/")
    def index():
        return FileResponse(DIST / "index.html")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        """SPA: serve file if exists, else index.html. Do not catch /api/* (handled by API routes)."""
        if full_path.startswith("api/"):
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Not found")
        file_path = DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(DIST / "index.html")
else:
    @app.get("/")
    def root():
        return {"message": "Build frontend first: cd frontend && npm run build"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
