# 🎬 BERT IMDB Sentiment — Streamlit Demo

## Project Structure
```
bert_demo/
├── app.py                  ← Main entry point
├── requirements.txt
└── pages/
    ├── home.py             ← Overview & pipeline
    ├── inference.py        ← Live sentiment prediction
    ├── hyperparam.py       ← Grid search results
    ├── regularization.py   ← Weight decay & dropout
    └── error_analysis.py   ← Confusion matrix, confidence, length, hard examples
```

## Setup & Run

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens at **http://localhost:8501**

## Point to Your Own Model

On the **Inference** page, expand ⚙️ Model Settings and paste your saved model path:
```
/content/drive/MyDrive/Colab Notebooks/bert-imdb-final
```

By default it uses `distilbert-base-uncased-finetuned-sst-2-english` from HuggingFace Hub as a demo.

## Pages

| Page | What it shows |
|---|---|
| 🏠 Home | Project pipeline summary |
| 🔮 Inference | Live sentiment prediction + temperature scaling |
| ⚙️ Hyperparameter Search | Heatmap + bar chart of grid search results |
| 🛡️ Regularization | Loss curves, weight decay, dropout comparison |
| 🔍 Error Analysis | Confusion matrix, confidence, length analysis, hard examples |
