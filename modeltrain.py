"""
ML pipeline: Random Forest, SVM, and Neural Network only.
Graphs and results show these three techniques only; best model is identified by accuracy.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Only these three techniques are used and displayed
TECHNIQUE_NAMES = ("Random Forest", "SVM", "Neural Network")

TARGET = "smoking"
ID_COL = "id"
N_FEATURES = 16
N_JOBS = -1
# Faster training: smaller models and subsamples
RF_ESTIMATORS = 100
RF_MAX_DEPTH = 16
SVM_CACHE = 512
SVM_SAMPLE = 15_000
NN_EPOCHS = 12
NN_BATCH = 512
NN_UNITS = (32, 16)
# Set to True for hyperparameter tuning (slower). False = use default params only.
ENABLE_TUNING = True


# ---------------------------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------------------------

def get_base_path():
    return Path(__file__).resolve().parent


def load_raw(base: Path) -> pd.DataFrame:
    """Load train.csv and return DataFrame. Strip column names for consistent fit/transform."""
    path = base / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Train file not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows with null/missing values and drop irrelevant columns.
    Returns DataFrame with only feature and target columns.
    """
    # Drop identifier (irrelevant for prediction)
    out = df.drop(columns=[ID_COL], errors="ignore")
    # Drop rows with any missing values
    out = out.dropna()
    return out


def select_features(X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, list, list, StandardScaler, SelectKBest]:
    """
    Select top K most important features by univariate F-score, then scale.
    Returns (X_scaled, feature_cols, fit_columns, scaler, selector).
    fit_columns = all columns selector was fit on (needed for transform on test).
    """
    n_k = min(N_FEATURES, X.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_k)
    X_selected = selector.fit_transform(X, y)
    cols = X.columns[selector.get_support()].tolist()
    fit_columns = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    return X_scaled.astype(np.float32), cols, fit_columns, scaler, selector


def load_and_prepare():
    """
    Load train data, clean (drop nulls, drop id), select top features, scale.
    Returns (X, y, feature_cols, scaler, selector) for training and persistence.
    """
    base = get_base_path()
    df = load_raw(base)
    df = clean_data(df)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    if X.isnull().any().any():
        raise ValueError("Unexpected nulls after clean_data (dropna).")

    X_scaled, feature_cols, fit_columns, scaler, selector = select_features(X, y)
    y = y.values.astype(np.int32)

    return X_scaled, y, feature_cols, scaler, selector, fit_columns


# ---------------------------------------------------------------------------
# 2. Train models: Random Forest, SVM, Neural Network only
# ---------------------------------------------------------------------------

def train_models(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
    """
    Train only Random Forest, SVM, and Neural Network.
    Returns list of (name, model, accuracy, is_keras) for these three techniques.
    """
    models = []

    # 1. Random Forest
    rf = RandomForestClassifier(
        n_estimators=RF_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        class_weight="balanced",
    )
    rf.fit(X_tr, y_tr)
    acc_rf = accuracy_score(y_val, rf.predict(X_val))
    models.append(("Random Forest", rf, acc_rf, False))

    # 2. Support Vector Machine (RBF kernel; data already scaled)
    n_svm = min(len(X_tr), SVM_SAMPLE)
    if n_svm < len(X_tr):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_tr), size=n_svm, replace=False)
        X_svm, y_svm = X_tr[idx], y_tr[idx]
    else:
        X_svm, y_svm = X_tr, y_tr
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        random_state=RANDOM_STATE,
        cache_size=SVM_CACHE,
        probability=True,
    )
    svm.fit(X_svm, y_svm)
    acc_svm = accuracy_score(y_val, svm.predict(X_val))
    models.append(("SVM", svm, acc_svm, False))

    # 3. Neural Network (if TensorFlow available)
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        tf.random.set_seed(RANDOM_STATE)
        nn = keras.Sequential([
            layers.Input(shape=(X_tr.shape[1],)),
            layers.Dense(NN_UNITS[0], activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(NN_UNITS[1], activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),
        ])
        nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        nn.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=NN_EPOCHS,
            batch_size=NN_BATCH,
            verbose=0,
        )
        pred_nn = (nn.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
        acc_nn = accuracy_score(y_val, pred_nn)
        models.append(("Neural Network", nn, acc_nn, True))
    except ImportError:
        pass

    return models


# ---------------------------------------------------------------------------
# 2b. Hyperparameter tuning for the best model
# ---------------------------------------------------------------------------

CV_FOLDS = 2
TUNE_N_ITER_RF = 6
TUNE_N_ITER_SVM = 5

def tune_best_model(
    best_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
) -> tuple:
    """
    Run RandomizedSearchCV for the given model type. Returns (tuned_model, tuned_accuracy, is_keras).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    if best_name == "Random Forest":
        param_dist = {
            "n_estimators": [80, 120, 150],
            "max_depth": [12, 16, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "class_weight": ["balanced"],
        }
        base = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=TUNE_N_ITER_RF,
            cv=cv,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=0,
        )
        search.fit(X_tr, y_tr)
        model = search.best_estimator_
        acc = accuracy_score(y_val, model.predict(X_val))
        return model, acc, False

    if best_name == "SVM":
        n_svm = min(len(X_tr), SVM_SAMPLE)
        if n_svm < len(X_tr):
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X_tr), size=n_svm, replace=False)
            X_fit, y_fit = X_tr[idx], y_tr[idx]
        else:
            X_fit, y_fit = X_tr, y_tr
        param_dist = {
            "C": [1.0, 2.0, 5.0],
            "gamma": ["scale", 0.01, 0.1],
            "kernel": ["rbf"],
        }
        base = SVC(random_state=RANDOM_STATE, cache_size=SVM_CACHE, probability=True)
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=TUNE_N_ITER_SVM,
            cv=cv,
            scoring="accuracy",
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=0,
        )
        search.fit(X_fit, y_fit)
        model = search.best_estimator_
        acc = accuracy_score(y_val, model.predict(X_val))
        return model, acc, False

    if best_name == "Neural Network":
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as e:
            print(f"  Tuning skipped (TensorFlow not available): {e}")
            return None, None, True

        # Manual grid over a few combinations (no scikeras required)
        best_acc = -1.0
        best_model = None
        configs = [
            {"u1": 32, "u2": 16, "dropout": 0.25, "epochs": 12, "batch": 512},
            {"u1": 48, "u2": 24, "dropout": 0.2, "epochs": 12, "batch": 512},
        ]
        for cfg in configs:
            tf.random.set_seed(RANDOM_STATE)
            m = keras.Sequential([
                layers.Input(shape=(n_features,)),
                layers.Dense(cfg["u1"], activation="relu"),
                layers.Dropout(cfg["dropout"]),
                layers.Dense(cfg["u2"], activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ])
            m.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
            m.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=cfg["epochs"], batch_size=cfg["batch"], verbose=0)
            pred = (m.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
            acc = accuracy_score(y_val, pred)
            if acc > best_acc:
                best_acc = acc
                best_model = m
        if best_model is not None:
            return best_model, best_acc, True
        return None, None, True
    return None, None, False


# ---------------------------------------------------------------------------
# 3. Save artifacts and report
# ---------------------------------------------------------------------------

def save_best_model(best_model, is_keras: bool, base: Path) -> None:
    """Save best model to best_model.joblib (and optionally .keras for Keras)."""
    path = base / "best_model.joblib"
    if is_keras:
        try:
            joblib.dump(best_model, path)
        except Exception:
            keras_path = base / "best_model.keras"
            best_model.save(keras_path)
            joblib.dump({"model_type": "keras", "path": str(keras_path)}, path)
    else:
        joblib.dump(best_model, path)
    print(f"Saved model to {path}")


def save_preprocessing(scaler: StandardScaler, feature_cols: list, selector: SelectKBest, base: Path) -> None:
    """Save scaler and feature list for inference (e.g. submission)."""
    path = base / "preprocessing.joblib"
    joblib.dump({
        "scaler": scaler,
        "feature_cols": feature_cols,
        "selector": selector,
    }, path)
    print(f"Saved preprocessing to {path}")


def write_results_json(
    models: list,
    best_name: str,
    best_acc: float,
    feature_cols: list,
    base: Path,
    plot_files: list[str] | None = None,
) -> None:
    """Write results.json for the app UI. Best = model with highest accuracy only."""
    path = base / "results.json"
    allowed = [m for m in models if m[0] in TECHNIQUE_NAMES]
    if allowed:
        best_entry = max(allowed, key=lambda m: m[2])
        best_name = best_entry[0]
        best_acc = best_entry[2]
    data = {
        "best_model": best_name if allowed else None,
        "best_accuracy": round(best_acc, 4) if allowed else 0,
        "models": [{"name": n, "accuracy": round(a, 4)} for n, _, a, _ in allowed],
        "available": True,
        "feature_cols": feature_cols,
        "plot_files": plot_files if plot_files is not None else [],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {path}")


def generate_submission(
    base: Path,
    best_model,
    is_keras: bool,
    feature_cols: list,
    scaler: StandardScaler,
    selector: SelectKBest,
    fit_columns: list,
) -> None:
    """
    Load test.csv, apply same preprocessing as train, predict with best model,
    write submission.csv (id, smoking) at project root for frontend/backend.
    fit_columns = all columns selector was fit on (must pass same columns to selector.transform).
    """
    test_path = base / "test.csv"
    if not test_path.exists():
        print("  test.csv not found; skipping submission.csv")
        return

    test_df = pd.read_csv(test_path)
    test_df.columns = test_df.columns.str.strip()
    if ID_COL not in test_df.columns:
        print("  test.csv missing 'id' column; skipping submission.csv")
        return

    missing = [c for c in fit_columns if c not in test_df.columns]
    if missing:
        print(f"  test.csv missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}; skipping submission.csv")
        return

    ids = test_df[ID_COL].values
    train_df = pd.read_csv(base / "train.csv")
    train_df.columns = train_df.columns.str.strip()
    for col in fit_columns:
        if test_df[col].isnull().any():
            med = train_df[col].median()
            test_df[col] = test_df[col].fillna(med)

    X_test = test_df[fit_columns]
    X_selected = selector.transform(X_test)
    X_scaled = scaler.transform(X_selected).astype(np.float32)

    # Predict: handle sklearn vs Keras
    if is_keras:
        try:
            # joblib may have saved the Keras model or a wrapper dict
            if hasattr(best_model, "predict"):
                proba = best_model.predict(X_scaled, verbose=0)
            else:
                import tensorflow as tf
                loaded = tf.keras.models.load_model(best_model.get("path"))
                proba = loaded.predict(X_scaled, verbose=0)
            pred = (np.array(proba).flatten() >= 0.5).astype(int)
        except Exception:
            pred = np.zeros(len(X_scaled), dtype=int)
    else:
        pred = best_model.predict(X_scaled)

    out = pd.DataFrame({ID_COL: ids, TARGET: pred})
    sub_path = base / "submission.csv"
    out.to_csv(sub_path, index=False)
    print(f"Saved {sub_path} ({len(out)} rows)")


def _get_proba_or_score(model, X: np.ndarray, is_keras: bool) -> np.ndarray:
    """Get prediction score for positive class (for ROC). Probabilities or decision function."""
    if is_keras:
        p = model.predict(X, verbose=0)
        return np.asarray(p).flatten()
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # Scale decision function to [0,1]-like for ROC (roc_curve accepts any score)
        d = model.decision_function(X)
        return (d - d.min()) / (d.max() - d.min() + 1e-9)
    # Fallback: use predict (0/1)
    return model.predict(X).astype(np.float64)


def generate_plots(
    models: list,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    base: Path,
    feature_cols: list[str] | None = None,
) -> list[str]:
    """
    Generate only the three model-result graphs: Random Forest, SVM, Neural Network.
    Returns list of saved filenames (model_comparison, confusion_matrices, roc_curves).
    """
    plots_dir = (base / "plots").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    # Only include the three techniques in every graph
    models = [m for m in models if m[0] in TECHNIQUE_NAMES]
    if not models:
        return []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, roc_curve, auc
    except ImportError as e:
        print(f"  matplotlib/sklearn not available: {e}; skipping plots")
        return []

    saved: list[str] = []

    def _save(fig, filename: str) -> None:
        path = plots_dir / filename
        fig.savefig(path, dpi=100, bbox_inches="tight", format="png")
        plt.close(fig)
        if path.is_file():
            saved.append(filename)
            print(f"  Saved {path}")

    # 1. model_comparison.png – Accuracy bar chart (highest accuracy first = best)
    sorted_models = sorted(models, key=lambda m: m[2], reverse=True)
    names = [m[0] for m in sorted_models]
    accs = [m[2] for m in sorted_models]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(names, accs, color="steelblue", edgecolor="navy", alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Technique")
    ax.set_title("Model Performance Comparison")
    fig.tight_layout()
    _save(fig, "model_comparison.png")

    # 2. confusion_matrices.png – One panel per technique
    if X_val is not None and y_val is not None:
        n = len(models)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        if n == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes).flatten()
        for i, (name, model, _, is_keras) in enumerate(models):
            ax = axes[i]
            pred = model.predict(X_val) if not is_keras else (
                (model.predict(X_val, verbose=0).flatten() >= 0.5).astype(int)
            )
            cm = confusion_matrix(y_val, pred)
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Pred 0", "Pred 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            for r in range(cm.shape[0]):
                for c in range(cm.shape[1]):
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center", color="black")
            ax.set_title(name)
        for j in range(len(models), len(axes)):
            axes[j].set_visible(False)
        fig.suptitle("Confusion Matrices (Validation Set)", fontsize=12, y=1.02)
        fig.tight_layout()
        _save(fig, "confusion_matrices.png")

    # 3. roc_curves.png – ROC for each of the three techniques
    if X_val is not None and y_val is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        for i, (name, model, _, is_keras) in enumerate(models):
            score = _get_proba_or_score(model, X_val, is_keras)
            fpr, tpr, _ = roc_curve(y_val, score)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{name} (AUC={roc_auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves – Model Comparison")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, "roc_curves.png")

    # 4. eda.png – EDA Training Data (target distribution)
    fig, ax = plt.subplots(figsize=(5, 3))
    unique, counts = np.unique(y_train, return_counts=True)
    ax.bar(unique.astype(str), counts, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_xlabel("Smoking (0=No, 1=Yes)")
    ax.set_title("EDA – Training Data")
    fig.tight_layout()
    _save(fig, "eda.png")

    # Load cleaned data for remaining EDA plots
    df_clean = None
    try:
        raw = load_raw(base)
        df_clean = clean_data(raw)
    except Exception as e:
        print(f"  Could not load train data for EDA: {e}")
        return saved

    if df_clean is None or TARGET not in df_clean.columns:
        return saved

    feat_cols = [c for c in df_clean.columns if c != TARGET]
    numeric_cols = df_clean[feat_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_candidates = [
        c for c in feat_cols
        if df_clean[c].nunique() <= 12 and c in df_clean.columns
    ]

    # 5. target_count.png – Target Variable Count Plot
    vc = df_clean[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(vc.index.astype(str), vc.values, color=["#2ecc71", "#e74c3c"], edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_xlabel("Smoking (0=No, 1=Yes)")
    ax.set_title("Target Variable Count Plot")
    fig.tight_layout()
    _save(fig, "target_count.png")

    # 6. histogram_numeric.png – Histogram (Numeric Features)
    if numeric_cols:
        n_hist = min(8, len(numeric_cols))
        cols_hist = numeric_cols[:n_hist]
        nrows, ncols = (2, 4) if n_hist >= 4 else (1, n_hist)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2.5 * nrows))
        axes = np.atleast_2d(axes)
        for i, col in enumerate(cols_hist):
            r, c = i // ncols, i % ncols
            axes[r, c].hist(df_clean[col].dropna(), bins=40, color="steelblue", edgecolor="white")
            axes[r, c].set_title(col[:18] + ".." if len(col) > 18 else col, fontsize=9)
        for i in range(len(cols_hist), axes.size):
            axes.flat[i].set_visible(False)
        fig.suptitle("Histogram (Numeric Features)", fontsize=12, y=1.02)
        fig.tight_layout()
        _save(fig, "histogram_numeric.png")

    # 7. box_plot.png – Box Plot (numeric features)
    if numeric_cols:
        n_box = min(8, len(numeric_cols))
        cols_box = numeric_cols[:n_box]
        fig, ax = plt.subplots(figsize=(max(8, n_box * 1.2), 5))
        df_clean[cols_box].boxplot(ax=ax)
        ax.set_ylabel("Value")
        ax.set_title("Box Plot")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        _save(fig, "box_plot.png")

    # 8. correlation_heatmap.png
    if len(numeric_cols) >= 2:
        try:
            import seaborn as sns
            corr = df_clean[numeric_cols + [TARGET]].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            _save(fig, "correlation_heatmap.png")
        except ImportError:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df_clean[numeric_cols + [TARGET]].corr()
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)
            plt.colorbar(im, ax=ax)
            ax.set_title("Correlation Heatmap")
            fig.tight_layout()
            _save(fig, "correlation_heatmap.png")

    # 9. bar_categorical_target.png – Bar Chart (Categorical vs Target)
    if categorical_candidates:
        n_cat = min(4, len(categorical_candidates))
        cols_cat = categorical_candidates[:n_cat]
        fig, axes = plt.subplots(2, 2, figsize=(9, 7))
        axes = axes.flatten()
        for i, col in enumerate(cols_cat):
            ax = axes[i]
            means = df_clean.groupby(col)[TARGET].mean()
            means.plot(kind="bar", ax=ax, color="steelblue", edgecolor="navy")
            ax.set_title(f"{col} vs Smoking rate")
            ax.set_ylabel("Mean smoking (1=Yes)")
            ax.tick_params(axis="x", rotation=45)
        for j in range(len(cols_cat), 4):
            axes[j].set_visible(False)
        fig.suptitle("Bar Chart (Categorical vs Target)", fontsize=12, y=1.02)
        fig.tight_layout()
        _save(fig, "bar_categorical_target.png")

    # 10. pair_plot.png – Pair Plot (Top 4–5 Important Features)
    top_features: list[str] = []
    if feature_cols and models:
        for _, model, _, is_keras in models:
            if not is_keras and hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                order = np.argsort(imp)[::-1]
                n_top = min(5, len(feature_cols))
                top_features = [feature_cols[j] for j in order[:n_top]]
                break
    if not top_features and numeric_cols:
        top_features = numeric_cols[:5]

    if top_features and all(c in df_clean.columns for c in top_features):
        plot_cols = top_features + [TARGET]
        sample_df = df_clean[plot_cols].sample(n=min(2000, len(df_clean)), random_state=RANDOM_STATE)
        try:
            import seaborn as sns
            g = sns.pairplot(sample_df, hue=TARGET, palette={0: "#2ecc71", 1: "#e74c3c"}, diag_kind="kde")
            g.fig.suptitle("Pair Plot (Top 4–5 Important Features)", y=1.02)
            path = plots_dir / "pair_plot.png"
            g.savefig(path, dpi=100, bbox_inches="tight", format="png")
            plt.close("all")
            if path.is_file():
                saved.append("pair_plot.png")
                print(f"  Saved {path}")
        except Exception as e:
            print(f"  pair_plot skipped: {e}")

    return saved


def main():
    base = get_base_path()

    print("Loading and cleaning data...")
    X, y, feature_cols, scaler, selector, fit_columns = load_and_prepare()
    n_total = len(y)
    print(f"  Rows after dropping nulls: {n_total}")
    print(f"  Selected features: {len(feature_cols)}")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  Train: {len(X_tr)}, Validation: {len(X_val)}")

    print("\nTraining models (Random Forest, SVM, Neural Network only)...")
    models = train_models(X_tr, y_tr, X_val, y_val)
    models = [m for m in models if m[0] in TECHNIQUE_NAMES]

    print("\n--- Accuracy ---")
    for name, _, acc, _ in models:
        print(f"  {name}: {acc:.4f}")

    best_name, best_model, best_acc, is_keras = max(models, key=lambda m: m[2])

    print("\n" + "=" * 50)
    print(f"BEST MODEL: {best_name} – {best_acc:.4f}")
    print("=" * 50)

    if ENABLE_TUNING:
        print(f"\nHyperparameter tuning for {best_name} (fast: 2-fold CV, few iterations)...")
        tuned_model, tuned_acc, tuned_is_keras = tune_best_model(
            best_name, X_tr, y_tr, X_val, y_val, X_tr.shape[1]
        )
        if tuned_model is not None and tuned_acc is not None:
            print(f"  Tuned validation accuracy: {tuned_acc:.4f} (was {best_acc:.4f})")
            models = [
                (n, (tuned_model if n == best_name else m), (tuned_acc if n == best_name else a), (tuned_is_keras if n == best_name else k))
                for n, m, a, k in models
            ]
            best_name, best_model, best_acc, is_keras = max(models, key=lambda m: m[2])
        else:
            print("  Tuning skipped or failed; using untuned best model.")
    else:
        print("  (Tuning disabled – set ENABLE_TUNING = True in pipeline.py for tuning)")

    print("\n" + "=" * 50)
    print(f"BEST MODEL (final): {best_name}")
    print(f"ACCURACY:  {best_acc:.4f}")
    if best_acc >= 0.85:
        print("Target (>= 85%) met.")
    else:
        print("Target (>= 85%) not met; consider more features or tuning.")
    print("=" * 50)

    save_best_model(best_model, is_keras, base)
    save_preprocessing(scaler, feature_cols, selector, base)

    print("\nGenerating submission.csv...")
    generate_submission(base, best_model, is_keras, feature_cols, scaler, selector, fit_columns)

    print("Generating plots...")
    plot_files = generate_plots(models, y, X_val, y_val, base, feature_cols=feature_cols)
    write_results_json(models, best_name, best_acc, feature_cols, base, plot_files=plot_files)


if __name__ == "__main__":
    main()
