<<<<<<< HEAD
# Kaggle Competition Pipeline – Distinction Criteria

This project implements a full ML pipeline for the smoking classification competition and a **single app run by browser** (traditional format: `app.py` at root, no `api` folder).

## Quick Start – Run by Browser

1. **Install and build frontend**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

2. **Run the app (API + UI served together)**
   ```bash
   pip install -r requirements.txt
   python app.py
   ```

3. **Open in browser:** http://localhost:8000

All UI and API run from this single process; no separate API folder.

## Project Structure (traditional format)

| File / folder | Purpose |
|---------------|---------|
| `app.py` | Single entry: FastAPI app – serves API and React frontend (run by browser). |
| `pipeline.py` | ML pipeline: load → EDA → clean → split → train → evaluate → submit → save model. Run for training and to generate `results.json`. |
| `train.csv`, `test.csv` | Data. |
| `results.json` | Written by `pipeline.py`; `app.py` serves it to the UI. |
| `submission.csv` | Generated predictions (after running pipeline). |
| `best_model.joblib` | Best-performing model. |
| `frontend/` | React UI (JSX). Build output in `frontend/dist` is served by `app.py`. |

## Run pipeline (training and report screenshots)

```bash
python pipeline.py
```

Plots open in windows; use them for **report screenshots**. After the run, refresh the browser to see model results and submission preview in the UI.

## Distinction Checklist Mapping

| Requirement | Where in Project |
|-------------|------------------|
| **EDA** | `run_eda()` in `pipeline.py`; EDA figure (target bar, age by class, feature correlations, heatmap). |
| **System architecture** | **Random Forest**, **SVM**, **Neural Network** in `pipeline.py`. Compare/contrast in your report. |
| **Model evaluation** | Accuracy, classification report, confusion matrix, ROC/AUC; plots and UI model comparison. |
| **Practical demo** | Screenshot of app in browser (http://localhost:8000) and/or pipeline plots. |
| **Conclusion** | In your own words: which model performed best and why. |

## Techniques Compared (for Report)

- **Random Forest**: Ensemble of trees; robust to scale; good baseline.
- **SVM (RBF)**: Margin-based; benefits from scaling; `probability=True` for ROC.
- **Neural Network**: Dense layers + dropout; learns non-linear boundaries.
=======
# Quitting-smoking
Assignment Model Train Submission
>>>>>>> e923fe7c031944c74208e0075b30ff6506255dbd
