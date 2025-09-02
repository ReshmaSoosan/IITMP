# Employee Attrition Prediction

## Project overview
Predict employee attrition using the IBM HR dataset. The notebook performs EDA, preprocessing, class-imbalance handling, trains several models, evaluates them, and saves the best model.

## Files
- `notebook.ipynb` — full analysis and modelling pipeline
- `report.pdf` — executive summary (dataset overview, model comparison, recommendations)
- `requirements.txt` — Python dependencies
- `model.pkl` — saved best model (pickle)
- `README.md` — how to run

## Requirements
```bash
pip install -r requirements.txt
```
## How to Run 

Open and run the notebook:
```bash
jupyter notebook notebook.ipynb
# or
jupyter lab
```

# Load model and use saved model
```python
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

preds = model.predict(X_new)
```
## Results & interpretation
See `report.pdf` for model metrics and recommended model. The main metrics to check: Accuracy, Precision, Recall, F1, ROC-AUC. Use the comparison table in the report.eport.o check: Accuracy, Precision, Recall, F1, ROC-AUC. Use the comparison table in the report.
