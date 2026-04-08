# money-guard

Money Guard: Online Payments Fraud Detection with Machine Learning

Production-style end-to-end project for detecting fraudulent online payment transactions using the dataset:

- `PS_20174392719_1491204439457_log.csv`

This repository includes:

- Data loading and validation
- Feature engineering pipeline
- Model training and threshold tuning
- Saved model artifacts and metrics
- FastAPI prediction service
- Unit tests and GitHub Actions CI

Dataset note:

- Keep `PS_20174392719_1491204439457_log.csv` in the project root locally.
- The dataset is excluded from Git (`.gitignore`) because it is too large for normal GitHub commits.

## 1) Project Structure

```text
.
|-- PS_20174392719_1491204439457_log.csv
|-- pyproject.toml
|-- requirements.txt
|-- scripts
|   |-- run_api.py
|   `-- train_model.py
|-- src
|   `-- fraud_detection
|       |-- __init__.py
|       |-- api.py
|       |-- config.py
|       |-- data.py
|       |-- features.py
|       |-- pipeline.py
|       |-- predict.py
|       `-- train.py
|-- tests
|   |-- test_features.py
|   `-- test_predict.py
`-- .github/workflows/ci.yml
```

## 2) Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
pip install -e .[dev]
```

## 3) Train the Model

Default training command (samples up to 500000 rows for speed):

```bash
python scripts/train_model.py
```

Custom data path / sampling:

```bash
python scripts/train_model.py --data-path PS_20174392719_1491204439457_log.csv --sample-frac 0.3 --sample-rows 300000
```

After training, files are created in `models/`:

- `fraud_model.joblib`
- `metrics.json`

## 4) Run the API

```bash
python scripts/run_api.py
```

API endpoints:

- `GET /health`
- `POST /predict`

Example prediction request:

```json
{
  "step": 1,
  "type": "TRANSFER",
  "amount": 181.0,
  "oldbalanceOrg": 181.0,
  "newbalanceOrig": 0.0,
  "oldbalanceDest": 0.0,
  "newbalanceDest": 0.0
}
```

Example using curl:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"step\":1,\"type\":\"TRANSFER\",\"amount\":181.0,\"oldbalanceOrg\":181.0,\"newbalanceOrig\":0.0,\"oldbalanceDest\":0.0,\"newbalanceDest\":0.0}"
```

## 5) Run Tests

```bash
pytest -q
```

## 6) Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: money-guard"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## 7) Notes

- The dataset is highly imbalanced. The training process optimizes a classification threshold from the validation set instead of using a fixed 0.5.
- Keep large model artifacts out of Git by default (configured in `.gitignore`).
- You can improve model quality later with hyperparameter tuning, class rebalancing strategies, or gradient boosting methods.
