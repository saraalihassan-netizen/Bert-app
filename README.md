import streamlit as st

st.set_page_config(
    page_title="BERT Sentiment Analysis — IMDB",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    st.caption("Model: bert-base-uncased  \nDataset: IMDB (50k reviews)  \nTask: Binary Sentiment")

if page == "🏠 Home":
    import pages.home as home
    home.show()
elif page == "🔮 Inference":
    import pages.inference as inference
    inference.show()
elif page == "⚙️ Hyperparameter Search":
    import pages.hyperparam as hyperparam
    hyperparam.show()
elif page == "🛡️ Regularization":
    import pages.regularization as regularization
    regularization.show()
elif page == "🔍 Error Analysis":
    import pages.error_analysis as error_analysis
    error_analysis.show()
