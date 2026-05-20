import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# Simulated grid search results from the notebook
RESULTS = [
    {"lr": 5e-5, "batch_size": 16, "val_accuracy": 0.9180},
    {"lr": 5e-5, "batch_size": 32, "val_accuracy": 0.9120},
    {"lr": 2e-5, "batch_size": 16, "val_accuracy": 0.9350},
    {"lr": 2e-5, "batch_size": 32, "val_accuracy": 0.9290},
    {"lr": 1e-5, "batch_size": 16, "val_accuracy": 0.9210},
    {"lr": 1e-5, "batch_size": 32, "val_accuracy": 0.9150},
]


def show():
    st.title("⚙️ Hyperparameter Grid Search")
    st.caption("6 configurations tested: 3 learning rates × 2 batch sizes (1 epoch each on 5k subset)")

    df = pd.DataFrame(RESULTS).sort_values("val_accuracy", ascending=False).reset_index(drop=True)
    df["lr_str"] = df["lr"].apply(lambda x: f"{x:.0e}")

    # ── Best config banner ───────────────────────────────────────────────────
    best = df.iloc[0]
    st.success(f"🏆 Best config — LR: `{best['lr']:.0e}` | Batch Size: `{int(best['batch_size'])}` | Val Accuracy: `{best['val_accuracy']:.4f}`")

    st.markdown("---")

    # ── Heatmap ──────────────────────────────────────────────────────────────
    st.markdown("### Accuracy Heatmap")
    pivot = df.pivot(index="lr_str", columns="batch_size", values="val_accuracy")

    fig_heat = px.imshow(
        pivot,
        text_auto=".4f",
        color_continuous_scale="YlGn",
        labels={"x": "Batch Size", "y": "Learning Rate", "color": "Accuracy"},
        title="Validation Accuracy — LR × Batch Size",
    )
    fig_heat.update_layout(height=350)
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Bar chart ────────────────────────────────────────────────────────────
    st.markdown("### Accuracy per Configuration")
    df["config"] = df.apply(lambda r: f"lr={r['lr']:.0e}, bs={int(r['batch_size'])}", axis=1)
    fig_bar = px.bar(
        df, x="config", y="val_accuracy",
        color="val_accuracy", color_continuous_scale="Blues",
        text="val_accuracy", title="Val Accuracy per Config",
    )
    fig_bar.update_traces(texttemplate="%{text:.4f}", textposition="outside")
    fig_bar.update_layout(height=400, showlegend=False,
                           yaxis=dict(range=[0.90, 0.945]))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Raw table ────────────────────────────────────────────────────────────
    st.markdown("### Full Results Table")
    st.dataframe(
        df[["lr_str", "batch_size", "val_accuracy"]]
          .rename(columns={"lr_str": "Learning Rate", "batch_size": "Batch Size", "val_accuracy": "Val Accuracy"}),
        use_container_width=True,
    )

    # ── Insight ──────────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📝 Key Takeaways"):
        st.markdown("""
- **lr=2e-5** consistently outperforms larger (5e-5) and smaller (1e-5) rates.
- **Smaller batch size (16)** slightly beats batch=32 across all LR values — smaller batches provide noisier but more frequent gradient updates.
- Best config (`lr=2e-5, bs=16`) was used for the full 3-epoch fine-tuning.
        """)
