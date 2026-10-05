# AI-Driven Behavioral Analysis of Phishing Websites Using Machine Learning

Phase 1 implements the reproducible ML foundation using the UCI PhiUSIIL Phishing URL (Website) dataset.

## Dataset
UCI PhiUSIIL:
https://archive.ics.uci.edu/dataset/967/phiusiil%2Bphishing%2Burl%2Bdataset

UCI reports 235,795 instances, 54 features, 134,850 legitimate URLs and 100,945 phishing URLs.
UCI defines label 1 = legitimate and label 0 = phishing.

## Phase 1 goals
1. Fetch the dataset through `ucimlrepo`.
2. Validate columns and labels.
3. Remove duplicate URLs before splitting.
4. Exclude identifier/raw categorical columns from the numeric baseline.
5. Create stratified 70/15/15 train/validation/test splits.
6. Train Logistic Regression, Random Forest, SVM and XGBoost.
7. Compare accuracy, precision, recall, F1, ROC-AUC, MCC and confusion matrices.
8. Save the selected model and metadata.
9. Produce feature importance and evaluation files.

## Important methodology
The dataset contains webpage-derived features as well as URL features. This phase is therefore a **benchmark/static webpage-feature model**, not yet the final live behavioral analyzer.

The final project will later add Playwright-based isolated browser analysis for:
- redirects
- DOM/HTML
- forms and form destinations
- JavaScript
- obfuscation
- iframes
- external resources
- network behavior
- brand similarity

Do not claim that Phase 1 is the full behavioral system.

## Run

### Windows
```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python src/train_phase1.py
```

### If the automatic UCI download fails
Download the CSV from the UCI page and place it at:
`data/raw/PhiUSIIL_Phishing_URL_Dataset.csv`

Then run:
```powershell
python src/train_phase1.py --csv data/raw/PhiUSIIL_Phishing_URL_Dataset.csv
```

Outputs:
- `data/processed/cleaned_dataset.csv`
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`
- `models/best_model.joblib`
- `models/model_metadata.json`
- `reports/model_comparison.csv`
- `reports/test_predictions.csv`
- `reports/confusion_matrix_<model>.png`
- `reports/feature_importance_<model>.csv`
- `reports/feature_importance_<model>.png`

## Label convention inside this project
For easier application logic:
- `0 = legitimate`
- `1 = phishing`

The source UCI dataset uses the opposite convention, so the script explicitly converts it.
