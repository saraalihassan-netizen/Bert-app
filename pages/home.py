import streamlit as st


def show():
    st.title("🎬 BERT Sentiment Analysis — IMDB")
    st.subheader("Deep Learning Lab | Error Analysis Project")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dataset", "IMDB", "50,000 reviews")
    col2.metric("Model", "BERT-base", "110M params")
    col3.metric("Best Accuracy", "93.5%", "+0.8% vs baseline")
    col4.metric("Best Temp (T)", "2.09", "calibrated")

    st.markdown("---")

    st.markdown("## 📌 Project Pipeline")

    steps = [
        ("1️⃣ Data & Tokenization", "Loaded IMDB dataset, tokenized with `BertTokenizer`, max_length=512."),
        ("2️⃣ Fine-Tuning BERT", "Trained `BertForSequenceClassification` for 3 epochs with HuggingFace Trainer."),
        ("3️⃣ Hyperparameter Search", "Grid search over learning rates [5e-5, 2e-5, 1e-5] × batch sizes [16, 32]."),
        ("4️⃣ Regularization", "Compared weight decay [0, 0.01, 0.1] and dropout [0.1, 0.3] effects."),
        ("5️⃣ Error Analysis", "Confusion matrix, confidence distributions, length-based error rate, temperature scaling."),
    ]

    for title, desc in steps:
        with st.expander(title, expanded=False):
            st.write(desc)

    st.markdown("---")
    st.info("👈 Use the sidebar to navigate between sections.")
