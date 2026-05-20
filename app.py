import streamlit as st

st.set_page_config(
    page_title="BERT Sentiment Analysis — IMDB",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ──────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎬 BERT IMDB Demo")
    st.caption("Deep Learning Lab — Error Analysis Project")
    st.divider()

    page = st.radio(
        "Navigate",
        options=[
            "🏠 Home",
            "🔮 Inference",
            "⚙️ Hyperparameter Search",
            "🛡️ Regularization",
            "🔍 Error Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Model: `bert-base-uncased`  \nDataset: IMDB (50k reviews)  \nTask: Binary Sentiment")

# ── Page routing ────────────────────────────────────────────────────────────
if page == "🏠 Home":
    from pages import home
    home.show()
elif page == "🔮 Inference":
    from pages import inference
    inference.show()
elif page == "⚙️ Hyperparameter Search":
    from pages import hyperparam
    hyperparam.show()
elif page == "🛡️ Regularization":
    from pages import regularization
    regularization.show()
elif page == "🔍 Error Analysis":
    from pages import error_analysis
    error_analysis.show()
