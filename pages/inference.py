import streamlit as st


# ── Helper: load model (cached so it loads once) ────────────────────────────
@st.cache_resource(show_spinner="Loading BERT model…")
def load_model(model_path: str):
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1,          # CPU; change to 0 for GPU
    )


def show():
    st.title("🔮 Live Sentiment Inference")
    st.caption("Enter any movie review and get an instant POSITIVE / NEGATIVE prediction.")

    # ── Model path config ────────────────────────────────────────────────────
    with st.expander("⚙️ Model Settings", expanded=False):
        model_path = st.text_input(
            "Model path or HuggingFace Hub ID",
            value="distilbert-base-uncased-finetuned-sst-2-english",
            help="Point to your saved bert-imdb-final folder, or use a public HF model as demo.",
        )

    st.markdown("---")

    # ── Sample reviews ───────────────────────────────────────────────────────
    samples = {
        "😃 Positive example": "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat throughout.",
        "😞 Negative example": "Terrible film. Worst thing I've seen in years. Complete waste of time and money — avoid at all costs.",
        "😐 Ambiguous example": "The movie had some great moments but also felt slow in places. Decent overall, nothing groundbreaking.",
    }

    st.markdown("#### Try a sample or write your own:")
    chosen = st.selectbox("Quick samples", ["— pick one —"] + list(samples.keys()))

    default_text = samples.get(chosen, "") if chosen != "— pick one —" else ""
    user_text = st.text_area(
        "Movie review",
        value=default_text,
        height=150,
        placeholder="Write or paste a movie review here…",
    )

    col_btn, col_temp = st.columns([1, 2])
    with col_btn:
        run = st.button("🚀 Analyse", use_container_width=True, type="primary")
    with col_temp:
        temperature = st.slider(
            "Temperature (calibration)",
            min_value=0.5, max_value=5.0, value=1.0, step=0.1,
            help="T=2.09 was the optimal calibration found in the notebook.",
        )

    # ── Prediction ───────────────────────────────────────────────────────────
    if run and user_text.strip():
        try:
            classifier = load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

        with st.spinner("Analysing…"):
            import torch, numpy as np
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            # Use pipeline for simplicity
            raw = classifier(user_text, return_all_scores=True)[0]

        # Apply temperature scaling to raw scores
        import numpy as np
        logits = np.array([s["score"] for s in raw])          # these are already softmax probs
        # Convert back to logits (approx), apply temp, re-softmax
        logits_approx = np.log(np.clip(logits, 1e-8, 1 - 1e-8))
        scaled = np.exp(logits_approx / temperature)
        calibrated = scaled / scaled.sum()

        label_map = {s["label"]: i for i, s in enumerate(raw)}
        labels = [s["label"] for s in raw]
        pred_idx = int(np.argmax(calibrated))
        pred_label = labels[pred_idx]
        confidence = float(calibrated[pred_idx])

        st.markdown("---")
        st.markdown("### 🎯 Result")

        c1, c2 = st.columns(2)
        if "POSITIVE" in pred_label or pred_label == "LABEL_1":
            c1.success(f"**POSITIVE** 😃")
        else:
            c1.error(f"**NEGATIVE** 😞")
        c2.metric("Confidence", f"{confidence:.1%}")

        # Score bars
        st.markdown("#### Score breakdown")
        for label, score in zip(labels, calibrated):
            nice = "POSITIVE 😃" if "POS" in label or label == "LABEL_1" else "NEGATIVE 😞"
            st.progress(float(score), text=f"{nice}: {score:.1%}")

        if temperature != 1.0:
            st.caption(f"ℹ️ Scores after temperature scaling (T={temperature:.1f}). High-confidence errors decreased by 38.5% at T=2.09.")

    elif run:
        st.warning("Please enter a review first.")
