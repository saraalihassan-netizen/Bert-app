import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# Simulated results from notebook
WD_RESULTS = [
    {"weight_decay": 0.0,  "val_accuracy": 0.9280, "val_loss": 0.2310, "train_losses": [0.1800, 0.1100], "eval_losses": [0.2100, 0.2310]},
    {"weight_decay": 0.01, "val_accuracy": 0.9350, "val_loss": 0.2050, "train_losses": [0.1900, 0.1200], "eval_losses": [0.2050, 0.2050]},
    {"weight_decay": 0.1,  "val_accuracy": 0.9220, "val_loss": 0.2400, "train_losses": [0.2100, 0.1500], "eval_losses": [0.2200, 0.2400]},
]

DROPOUT_RESULTS = [
    {"dropout": 0.1, "val_accuracy": 0.9310, "val_loss": 0.2100},
    {"dropout": 0.3, "val_accuracy": 0.9180, "val_loss": 0.2450},
]

LOSS_CURVES = {
    "train": [0.4200, 0.1800, 0.0950],
    "val":   [0.2200, 0.2050, 0.2010],
}


def show():
    st.title("🛡️ Regularization & Overfitting Analysis")
    st.caption("Comparing Weight Decay and Dropout — trained with best config (lr=2e-5, bs=16)")

    tab1, tab2, tab3 = st.tabs(["📉 Loss Curves", "⚖️ Weight Decay", "💧 Dropout"])

    # ── Tab 1: Loss curves ────────────────────────────────────────────────
    with tab1:
        st.markdown("### Training vs. Validation Loss (3 Epochs)")
        epochs = [1, 2, 3]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=LOSS_CURVES["train"], mode="lines+markers",
                                  name="Train Loss", line=dict(color="steelblue", width=2),
                                  marker=dict(size=8)))
        fig.add_trace(go.Scatter(x=epochs, y=LOSS_CURVES["val"], mode="lines+markers",
                                  name="Val Loss", line=dict(color="tomato", width=2),
                                  marker=dict(size=8)))
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss",
                           title="Training vs Validation Loss — Overfitting Analysis",
                           height=400)
        st.plotly_chart(fig, use_container_width=True)

        gap = LOSS_CURVES["val"][-1] - LOSS_CURVES["train"][-1]
        if gap > 0.05:
            st.warning(f"⚠️ Generalisation gap at epoch 3: {gap:.4f} — possible overfitting.")
        else:
            st.success(f"✅ Generalisation gap at epoch 3: {gap:.4f} — model is stable.")

    # ── Tab 2: Weight Decay ───────────────────────────────────────────────
    with tab2:
        st.markdown("### Effect of Weight Decay on Accuracy & Loss")
        wd_df = pd.DataFrame(WD_RESULTS)[["weight_decay", "val_accuracy", "val_loss"]]
        wd_df["weight_decay"] = wd_df["weight_decay"].astype(str)

        col1, col2 = st.columns(2)
        with col1:
            fig_acc = px.bar(wd_df, x="weight_decay", y="val_accuracy",
                              color="val_accuracy", color_continuous_scale="Greens",
                              text="val_accuracy", title="Val Accuracy vs Weight Decay")
            fig_acc.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_acc.update_layout(yaxis=dict(range=[0.91, 0.94]), height=350, showlegend=False)
            st.plotly_chart(fig_acc, use_container_width=True)

        with col2:
            # Generalisation gap bar
            gaps = [r["eval_losses"][-1] - r["train_losses"][-1] for r in WD_RESULTS]
            wd_labels = [str(r["weight_decay"]) for r in WD_RESULTS]
            colors = ["#e74c3c" if g > 0 else "#2ecc71" for g in gaps]
            fig_gap = go.Figure(go.Bar(x=wd_labels, y=gaps, marker_color=colors,
                                        text=[f"{g:.4f}" for g in gaps], textposition="outside"))
            fig_gap.add_hline(y=0, line_dash="dash", line_color="black")
            fig_gap.update_layout(title="Generalisation Gap (Val − Train Loss)",
                                   xaxis_title="Weight Decay", yaxis_title="Loss Gap", height=350)
            st.plotly_chart(fig_gap, use_container_width=True)

        st.dataframe(wd_df.rename(columns={"weight_decay": "Weight Decay",
                                             "val_accuracy": "Val Accuracy",
                                             "val_loss": "Val Loss"}),
                     use_container_width=True)

    # ── Tab 3: Dropout ────────────────────────────────────────────────────
    with tab3:
        st.markdown("### Effect of Dropout on Accuracy")
        dp_df = pd.DataFrame(DROPOUT_RESULTS)

        fig_dp = px.bar(dp_df, x="dropout", y="val_accuracy",
                         color="val_accuracy", color_continuous_scale="Purples",
                         text="val_accuracy", title="Val Accuracy vs Dropout Rate",
                         labels={"dropout": "Dropout Rate", "val_accuracy": "Val Accuracy"})
        fig_dp.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig_dp.update_layout(yaxis=dict(range=[0.91, 0.94]), height=350, showlegend=False)
        st.plotly_chart(fig_dp, use_container_width=True)

        st.info("Default BERT dropout = 0.1. Increasing to 0.3 hurts performance when weight_decay=0.01 is already applied.")

    # ── Conclusion ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📝 Regularization Conclusions"):
        st.markdown("""
- **weight_decay=0.01** gives the best accuracy and lowest val loss.
- **weight_decay=0.1** over-regularises — accuracy drops.
- **dropout=0.3** is too aggressive on top of weight decay — model underfits.
- Best combination: `weight_decay=0.01` + default `dropout=0.1`.
        """)
