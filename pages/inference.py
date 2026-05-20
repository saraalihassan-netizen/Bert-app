import streamlit as st


@st.cache_resource(show_spinner="Loading model…")
def load_model(model_path: str):
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=model_path,
        tokenizer=model_path,
        device=-1,
    )


def show():
    st.title("🔮 Live Sentiment Inference")
    st.caption("Enter any movie review and get an instant POSITIVE / NEGATIVE prediction.")

    with st.expander("⚙️ Model Settings", expanded=False):
        model_path = st.text_input(
            "Model path or HuggingFace Hub ID",
            value="distilbert-base-uncased-finetuned-sst-2-english",
        )

    st.markdown("---")

    samples = {
        "😃 Positive example": "This movie was absolutely fantastic! The acting was superb and the plot kept me on the edge of my seat throughout.",
        "😞 Negative example": "Terrible film. Worst thing I've seen in years. Complete waste of time and money.",
        "😐 Ambiguous example": "The movie had some great moments but also felt slow in places. Decent overall.",
    }

    chosen = st.selectbox("Quick samples", ["— pick one —"] + list(samples.keys()))
    default_text = samples.get(chosen, "") if chosen != "— pick one —" else ""

    user_text = st.text_area("Movie review", value=default_text, height=150,
                              placeholder="Write or paste a movie review here…")

    col_btn, col_temp = st.columns([1, 2])
    with col_btn:
        run = st.button("🚀 Analyse", use_container_width=True, type="primary")
    with col_temp:
        temperature = st.slider("Temperature (calibration)",
                                min_value=0.5, max_value=5.0, value=1.0, step=0.1)

    if run and user_text.strip():
        try:
            classifier = load_model(model_path)
        except Exception as e:
            st.error(f"Could not load model: {e}")
            st.stop()

        with st.spinner("Analysing…"):
            results = classifier(user_text, top_k=None)
        import numpy as np

        labels = [r["label"] for r in results]
        scores = np.array([r["score"] for r in results])

        logits = np.log(np.clip(scores, 1e-8, 1 - 1e-8))
        scaled = np.exp(logits / temperature)
        calibrated = scaled / scaled.sum()

        pred_idx = int(np.argmax(calibrated))
        pred_label = labels[pred_idx]
        confidence = float(calibrated[pred_idx])

        st.markdown("---")
        st.markdown("### 🎯 Result")

        c1, c2 = st.columns(2)
        if "POSITIVE" in pred_label or pred_label == "LABEL_1":
            c1.success("*POSITIVE* 😃")
        else:
            c1.error("*NEGATIVE* 😞")
        c2.metric("Confidence", f"{confidence:.1%}")

        st.markdown("#### Score breakdown")
        for label, score in zip(labels, calibrated):
            nice = "POSITIVE 😃" if "POS" in label or label == "LABEL_1" else "NEGATIVE 😞"
            st.progress(float(score), text=f"{nice}: {score:.1%}")

    elif run:
        st.warning("Please enter a review first.")
