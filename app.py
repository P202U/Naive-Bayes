import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Naive Bayes Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 60%, #0a1020 100%);
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #091322 100%);
    border-right: 1px solid #1e3a5f;
}
h1 { font-family:'Space Mono',monospace !important; color:#38bdf8 !important; }
h2 { color:#7dd3fc !important; }
h3 { color:#93c5fd !important; }

.stTabs [data-baseweb="tab-list"] {
    background: rgba(14,30,54,0.8);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid #1e3a5f;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: #64748b;
    border-radius: 8px;
    padding: 8px 14px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#0369a1,#0284c7) !important;
    color: white !important;
}

.card {
    background: rgba(14,30,54,0.85);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.metric-card {
    background: rgba(14,30,54,0.9);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value { font-family:'Space Mono',monospace; font-size:2rem; color:#38bdf8; font-weight:700; }
.metric-label { font-size:0.72rem; color:#64748b; text-transform:uppercase; letter-spacing:.1em; }

.pred-box {
    text-align:center; padding:1.6rem; border-radius:14px; margin:1rem 0;
    font-family:'Space Mono',monospace; font-size:1.6rem; font-weight:700;
    letter-spacing:.05em; box-shadow:0 0 40px rgba(56,189,248,.15);
}
.pred-yes { background:linear-gradient(135deg,#064e3b,#065f46); border:2px solid #10b981; color:#6ee7b7; }
.pred-no  { background:linear-gradient(135deg,#450a0a,#7f1d1d); border:2px solid #ef4444; color:#fca5a5; }
.pred-neutral { background:linear-gradient(135deg,#1e1b4b,#312e81); border:2px solid #6366f1; color:#a5b4fc; }

.tug-bar-wrap {
    background: #0d1526;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
}
.tug-label { font-family:'Space Mono',monospace; font-size:.78rem; color:#94a3b8; margin-bottom:4px; }
.tug-bar-bg {
    height: 22px; border-radius: 6px; overflow: hidden;
    background: linear-gradient(90deg, #7f1d1d 0%, #1e3a5f 50%, #064e3b 100%);
    position: relative;
}
.tug-marker {
    position: absolute; top: 0; height: 100%; width: 4px;
    background: white; border-radius: 2px;
    box-shadow: 0 0 8px white;
}
.tug-center { position:absolute; top:0; left:50%; width:2px; height:100%; background:#475569; }

.stButton > button {
    background: linear-gradient(135deg,#0369a1,#0284c7) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family:'Space Mono',monospace !important;
    font-weight:700 !important; letter-spacing:.05em !important;
    padding:.6rem 1.5rem !important; transition: all .2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg,#0284c7,#38bdf8) !important;
    box-shadow: 0 0 20px rgba(56,189,248,.4) !important;
    transform: translateY(-1px) !important;
}
hr { border-color:#1e3a5f !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def is_binary_col(s):
    u = set(s.dropna().astype(str).str.strip().str.lower())
    return u.issubset({"yes", "no", "true", "false", "1", "0", "y", "n"})


def norm_binary(s):
    m = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
    return s.astype(str).str.strip().str.lower().map(m)


def encode_df(df, feature_cols, target_col):
    X_enc = pd.DataFrame()
    encoders = {}
    for col in feature_cols:
        if is_binary_col(df[col]):
            X_enc[col] = norm_binary(df[col])
        else:
            le = LabelEncoder()
            X_enc[col] = le.fit_transform(df[col].astype(str).str.strip().str.lower())
            encoders[col] = le

    if is_binary_col(df[target_col]):
        y = norm_binary(df[target_col]).values.astype(int)
        target_classes = ["No", "Yes"]
        target_binary = True
    else:
        le = LabelEncoder()
        y = le.fit_transform(df[target_col].astype(str).str.strip().str.lower())
        target_classes = list(le.classes_)
        encoders["__target__"] = le
        target_binary = False

    return X_enc.values.astype(int), y, encoders, target_classes, target_binary


def do_train(df, feature_cols, target_col):
    X, y, encoders, target_classes, target_binary = encode_df(
        df, feature_cols, target_col
    )
    model = BernoulliNB()
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    return model, encoders, target_classes, target_binary, X, y, acc


def encode_input(input_dict, feature_cols, encoders):
    row = []
    for col in feature_cols:
        val = str(input_dict[col]).strip().lower()
        if col in encoders:
            row.append(int(encoders[col].transform([val])[0]))
        else:
            m = {
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "y": 1,
                "n": 0,
            }
            row.append(m.get(val, 0))
    return np.array([row])


def fig_style(fig, axes=None):
    fig.patch.set_facecolor("#0a0e1a")
    if axes is None:
        return
    ax_list = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    for ax in ax_list:
        ax.set_facecolor("#0d1526")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#7dd3fc")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e3a5f")


def feature_input_widgets(feat_cols, df_source, key_prefix):
    """Render radio/select widgets for features; return input_dict."""
    input_dict = {}
    groups = [feat_cols[i : i + 3] for i in range(0, len(feat_cols), 3)]
    for grp in groups:
        cols = st.columns(len(grp))
        for i, cn in enumerate(grp):
            with cols[i]:
                uv = sorted(df_source[cn].dropna().astype(str).str.strip().unique())
                if is_binary_col(df_source[cn]):
                    input_dict[cn] = st.radio(
                        f"**{cn}**", uv, horizontal=True, key=f"{key_prefix}_{cn}"
                    )
                else:
                    input_dict[cn] = st.selectbox(
                        f"**{cn}**", uv, key=f"{key_prefix}_{cn}"
                    )
    return input_dict


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧪 NB Lab")
    st.markdown("---")
    st.markdown(
        """
**Five tabs:**

🔬 **Train** — upload CSV & train  
🎛️ **Tug-of-War** — belief shift per feature  
📊 **Feature Importance** — log-likelihood bars  
🧬 **Synthetic Lab** — generate data & test sensitivity  
🗺️ **Decision Boundary** — 2-D class map  
    """
    )
    st.markdown("---")
    example = """Cough,Fever,Headache,Runny Nose,Flu
Yes,Yes,Yes,Yes,Yes
Yes,Yes,No,Yes,Yes
Yes,No,Yes,No,Yes
No,Yes,Yes,Yes,Yes
Yes,Yes,No,No,Yes
No,No,Yes,No,No
No,Yes,No,No,No
No,No,No,No,No
Yes,No,No,No,No
No,No,Yes,Yes,No"""
    st.download_button("📥 Example CSV", example, "flu_data.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
#  TITLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🧪 Naive Bayes Lab")
st.markdown("*Train · Simulate · Understand — four ways to feel the math.*")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔬 Train",
        "🎛️ Tug-of-War",
        "📊 Feature Importance",
        "🧬 Synthetic Lab",
        "🗺️ Decision Boundary",
    ]
)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### 📂 Upload Dataset")
    uploaded = st.file_uploader("Drop a CSV here", type=["csv"], key="main_upload")

    if uploaded:
        df = pd.read_csv(uploaded)
        df.columns = df.columns.str.strip()
        st.success(f"✅ {len(df)} rows × {len(df.columns)} columns")
        with st.expander("👁️ Preview", expanded=False):
            st.dataframe(df, use_container_width=True, height=200)

        st.markdown("---")
        st.markdown("### ⚙️ Configure")
        c1, c2 = st.columns([2, 1])
        with c1:
            feat_cols = st.multiselect(
                "🔷 Feature Columns",
                df.columns.tolist(),
                default=df.columns.tolist()[:-1],
            )
        with c2:
            remaining = [c for c in df.columns if c not in feat_cols]
            tgt_col = st.selectbox(
                "🎯 Target Column", remaining if remaining else df.columns.tolist()
            )

        if feat_cols and st.button("🚀 Train Model", key="train_btn"):
            with st.spinner("Training..."):
                try:
                    model, encoders, target_classes, target_binary, X, y, acc = (
                        do_train(df, feat_cols, tgt_col)
                    )
                    st.session_state.update(
                        {
                            "model": model,
                            "encoders": encoders,
                            "target_classes": target_classes,
                            "target_binary": target_binary,
                            "feat_cols": feat_cols,
                            "tgt_col": tgt_col,
                            "acc": acc,
                            "X": X,
                            "y": y,
                            "df": df,
                        }
                    )
                    st.success("✅ Model trained! Switch to any tab to explore.")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    if "model" in st.session_state:
        model = st.session_state["model"]
        target_classes = st.session_state["target_classes"]
        feat_cols_s = st.session_state["feat_cols"]
        acc_s = st.session_state["acc"]
        X_s = st.session_state["X"]
        y_s = st.session_state["y"]

        st.markdown("---")
        st.markdown("### 📊 Model Stats")
        c1, c2, c3, c4 = st.columns(4)
        for col, val, lbl in zip(
            [c1, c2, c3, c4],
            [
                f"{acc_s*100:.1f}%",
                len(st.session_state["df"]),
                len(feat_cols_s),
                len(target_classes),
            ],
            ["Train Accuracy", "Samples", "Features", "Classes"],
        ):
            with col:
                st.markdown(
                    f"""
                <div class="metric-card">
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>""",
                    unsafe_allow_html=True,
                )

        st.markdown("")
        y_pred = model.predict(X_s)
        cm = confusion_matrix(y_s, y_pred)
        labels = (
            target_classes
            if len(target_classes) == cm.shape[0]
            else [str(i) for i in range(cm.shape[0])]
        )

        fig, ax = plt.subplots(figsize=(5, 4))
        fig_style(fig, ax)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            linewidths=0.5,
            linecolor="#1e3a5f",
        )
        ax.set_title("Confusion Matrix", fontsize=13)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("📋 Full Classification Report"):
            rpt = classification_report(
                y_s, y_pred, target_names=labels, output_dict=True
            )
            st.dataframe(pd.DataFrame(rpt).T.round(3), use_container_width=True)
    else:
        st.info("Upload a CSV above and press Train to get started.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — TUG-OF-WAR
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if "model" not in st.session_state:
        st.info("⬅️ Train a model in the **Train** tab first.")
    else:
        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        target_classes = st.session_state["target_classes"]
        feat_cols = st.session_state["feat_cols"]
        tgt_col = st.session_state["tgt_col"]
        df_s = st.session_state["df"]
        n_classes = len(target_classes)

        st.markdown("### 🎛️ Set Feature Values")
        input_dict = feature_input_widgets(feat_cols, df_s, "tw")

        st.markdown("---")
        st.markdown("### ⚖️ Probability Tug-of-War")
        st.markdown(
            "*Each row below shows the cumulative belief **after** that feature is added. "
            "Watch the marker drift left (No) or right (Yes).*"
        )

        # Step-by-step belief
        log_prior = model.class_log_prior_.copy()  # (n_classes,)
        prior_probs = np.exp(log_prior)
        prior_probs /= prior_probs.sum()

        X_new = encode_input(input_dict, feat_cols, encoders)
        cumulative_log = log_prior.copy()
        steps = [("Prior (before any evidence)", prior_probs.copy())]

        for fi, cn in enumerate(feat_cols):
            feat_val = int(X_new[0, fi])
            for ci in range(n_classes):
                p1 = model.feature_log_prob_[ci, fi]
                p0 = np.log(max(1 - np.exp(p1), 1e-10))
                cumulative_log[ci] += p1 if feat_val == 1 else p0
            step_probs = np.exp(cumulative_log - cumulative_log.max())
            step_probs /= step_probs.sum()
            val_display = input_dict[cn]
            steps.append((f"{cn} = {val_display}", step_probs.copy()))

        if n_classes == 2:
            for step_name, probs in steps:
                p_yes = float(probs[1])
                bar_pct = int(p_yes * 96) + 2  # keep inside bar
                is_prior = step_name.startswith("Prior")
                color = "#10b981" if p_yes >= 0.5 else "#ef4444"
                prefix = "📌" if is_prior else ("✅" if p_yes >= 0.5 else "❌")
                st.markdown(
                    f"""
                <div class="tug-bar-wrap">
                  <div class="tug-label">
                    {prefix} &nbsp;<b style="color:#e2e8f0">{step_name}</b>
                    &nbsp;→&nbsp;
                    <span style="color:{color};font-weight:700">
                      P({target_classes[1]}) = {p_yes*100:.1f}%
                    </span>
                  </div>
                  <div class="tug-bar-bg">
                    <div class="tug-center"></div>
                    <div class="tug-marker" style="left:{bar_pct}%"></div>
                  </div>
                  <div style="display:flex;justify-content:space-between;
                              font-family:'Space Mono',monospace;font-size:.65rem;
                              color:#475569;margin-top:3px">
                    <span>← {target_classes[0]}</span>
                    <span>{target_classes[1]} →</span>
                  </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
        else:
            for step_name, probs in steps:
                best = int(np.argmax(probs))
                lbl = target_classes[best] if best < len(target_classes) else str(best)
                row = {tc: f"{p*100:.1f}%" for tc, p in zip(target_classes, probs)}
                st.markdown(
                    f"**{step_name}** → leading: `{lbl}` ({probs[best]*100:.1f}%)"
                )
                st.dataframe(
                    pd.DataFrame([row]), use_container_width=True, hide_index=True
                )

        # Final box
        final_probs = steps[-1][1]
        pred_idx = int(np.argmax(final_probs))
        pred_label = (
            target_classes[pred_idx]
            if pred_idx < len(target_classes)
            else str(pred_idx)
        )
        is_pos = pred_label.lower() in ["yes", "true", "1"]
        is_neg = pred_label.lower() in ["no", "false", "0"]
        box_cls = "pred-yes" if is_pos else ("pred-no" if is_neg else "pred-neutral")
        icon = "✅" if is_pos else ("❌" if is_neg else "🔵")
        st.markdown(
            f"""
        <div class="pred-box {box_cls}">
          {icon} &nbsp; {tgt_col}: <span style="text-transform:uppercase">{pred_label}</span>
          &nbsp;·&nbsp; {final_probs[pred_idx]*100:.1f}% confidence
        </div>""",
            unsafe_allow_html=True,
        )

        # Mini Bayesian maths recap
        with st.expander("🧮 The maths behind the bars"):
            st.markdown(
                f"""
**Bayes' theorem — step by step:**

1. Start with **Prior** → P(Flu=Yes) from the training data base rate.
2. For each feature you set, multiply by the **Likelihood**:

   &nbsp;&nbsp;&nbsp; `P(Flu=Yes | all features) ∝ P(Flu=Yes) × ∏ P(featureᵢ | Flu=Yes)`

3. The bar marker moves right if `P(feature | Yes) > P(feature | No)`, left otherwise.
4. The final marker position = the model's posterior belief after seeing **all** your evidence.

The key word is **Naive**: each feature pulls independently — they never "talk" to each other.
            """
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    if "model" not in st.session_state:
        st.info("⬅️ Train a model in the **Train** tab first.")
    else:
        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        target_classes = st.session_state["target_classes"]
        feat_cols = st.session_state["feat_cols"]
        tgt_col = st.session_state["tgt_col"]
        df_s = st.session_state["df"]
        n_classes = len(target_classes)

        st.markdown("### 📊 Feature Importance — Log-Likelihood Ratio")
        st.markdown(
            """
*Each bar shows how much a feature **supports (+)** or **opposes (−)** the predicted class.*  
`score = log P(feat_value | predicted_class) − avg log P(feat_value | other_classes)`
        """
        )

        st.markdown("#### Set inputs:")
        input_dict3 = feature_input_widgets(feat_cols, df_s, "fi")

        X_new3 = encode_input(input_dict3, feat_cols, encoders)
        proba3 = model.predict_proba(X_new3)[0]
        pred3 = int(np.argmax(proba3))
        pred_label3 = (
            target_classes[pred3] if pred3 < len(target_classes) else str(pred3)
        )

        # Compute log-likelihood ratio per feature
        ll_scores = []
        for fi, cn in enumerate(feat_cols):
            feat_val = int(X_new3[0, fi])
            p1_pred = model.feature_log_prob_[pred3, fi]
            p0_pred = np.log(max(1 - np.exp(p1_pred), 1e-10))
            ll_pred = p1_pred if feat_val == 1 else p0_pred

            other_cls = [ci for ci in range(n_classes) if ci != pred3]
            if other_cls:
                ll_others_vals = []
                for ci in other_cls:
                    p1_o = model.feature_log_prob_[ci, fi]
                    p0_o = np.log(max(1 - np.exp(p1_o), 1e-10))
                    ll_others_vals.append(p1_o if feat_val == 1 else p0_o)
                ll_ratio = ll_pred - np.mean(ll_others_vals)
            else:
                ll_ratio = ll_pred

            ll_scores.append((cn, ll_ratio, input_dict3[cn]))

        ll_scores.sort(key=lambda x: x[1])
        names = [f"{x[0]} = {x[2]}" for x in ll_scores]
        values = [x[1] for x in ll_scores]
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in values]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)))
        fig_style(fig, ax)
        bars = ax.barh(names, values, color=colors, edgecolor="#1e3a5f", linewidth=0.5)
        ax.axvline(0, color="#475569", linewidth=1.2, linestyle="--")
        ax.set_xlabel("Log-Likelihood Ratio  (+ supports  |  − opposes)")
        ax.set_title(f'Why the model predicts  "{pred_label3}"', fontsize=12)

        for bar, val in zip(bars, values):
            sign = "+" if val >= 0 else ""
            ax.text(
                val + (0.02 if val >= 0 else -0.02),
                bar.get_y() + bar.get_height() / 2,
                f"{sign}{val:.2f}",
                va="center",
                ha="left" if val >= 0 else "right",
                color="#e2e8f0",
                fontsize=8,
                fontfamily="monospace",
            )

        green_p = mpatches.Patch(color="#10b981", label="Supports prediction")
        red_p = mpatches.Patch(color="#ef4444", label="Opposes prediction")
        ax.legend(
            handles=[green_p, red_p],
            facecolor="#0d1526",
            edgecolor="#1e3a5f",
            labelcolor="#e2e8f0",
            fontsize=8,
        )
        plt.tight_layout()
        st.pyplot(fig)

        # Final prediction
        is_pos3 = pred_label3.lower() in ["yes", "true", "1"]
        is_neg3 = pred_label3.lower() in ["no", "false", "0"]
        box3 = "pred-yes" if is_pos3 else ("pred-no" if is_neg3 else "pred-neutral")
        icon3 = "✅" if is_pos3 else ("❌" if is_neg3 else "🔵")
        st.markdown(
            f"""
        <div class="pred-box {box3}" style="font-size:1.1rem;padding:1rem">
          {icon3} &nbsp; {tgt_col}: <b>{pred_label3}</b>
          &nbsp;·&nbsp; confidence {proba3[pred3]*100:.1f}%
        </div>""",
            unsafe_allow_html=True,
        )

        st.markdown("#### 📋 Score Table")
        tbl = pd.DataFrame(
            {
                "Feature": [x[0] for x in ll_scores[::-1]],
                "Value Set": [x[2] for x in ll_scores[::-1]],
                "Log-Likelihood Ratio": [f"{x[1]:+.4f}" for x in ll_scores[::-1]],
                "Verdict": [
                    "✅ Supports" if x[1] >= 0 else "❌ Opposes"
                    for x in ll_scores[::-1]
                ],
            }
        )
        st.dataframe(tbl, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — SYNTHETIC DATA LAB
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🧬 Synthetic Data Generator")
    st.markdown(
        """
*Design a dataset from scratch. Dial in how much each feature correlates with the target,
add noise, and watch the model's confusion matrix & accuracy shift in real time.*
    """
    )

    col_l, col_r = st.columns([1, 1.6])

    with col_l:
        st.markdown("#### ⚙️ Lab Controls")
        n_samples = st.slider("Samples", 50, 2000, 300, 50, key="syn_n")
        n_features = st.slider("Features", 2, 8, 3, key="syn_f")
        noise_pct = st.slider("🌀 Noise (%)", 0, 60, 15, key="syn_noise")

        st.markdown("**Feature → Target strength:**")
        feat_strengths = []
        feat_names_syn = []
        for fi in range(n_features):
            name = f"Feature_{fi+1}"
            s = st.slider(f"  {name}", 10, 99, 70, key=f"fs_{fi}")
            feat_names_syn.append(name)
            feat_strengths.append(s / 100)

        gen_btn = st.button("⚡ Generate & Train", key="synth_btn")

    with col_r:
        if gen_btn:
            rng = np.random.default_rng(42)
            y_syn = rng.integers(0, 2, n_samples)
            X_syn = np.zeros((n_samples, n_features), dtype=int)
            for fi, strength in enumerate(feat_strengths):
                for ri in range(n_samples):
                    X_syn[ri, fi] = (
                        y_syn[ri] if rng.random() < strength else 1 - y_syn[ri]
                    )
            # noise
            noise_mask = rng.random(X_syn.shape) < (noise_pct / 100)
            X_syn[noise_mask] = 1 - X_syn[noise_mask]

            mdl_syn = BernoulliNB()
            mdl_syn.fit(X_syn, y_syn)
            y_pred_syn = mdl_syn.predict(X_syn)
            acc_syn = accuracy_score(y_syn, y_pred_syn)
            cm_syn = confusion_matrix(y_syn, y_pred_syn)

            st.session_state.update(
                {
                    "syn_model": mdl_syn,
                    "syn_X": X_syn,
                    "syn_y": y_syn,
                    "syn_feat_names": feat_names_syn,
                    "syn_acc": acc_syn,
                    "syn_cm": cm_syn,
                    "syn_ns": n_samples,
                    "syn_noise_val": noise_pct,
                    "syn_strengths": feat_strengths,
                }
            )

        if "syn_model" in st.session_state:
            acc_syn = st.session_state["syn_acc"]
            cm_syn = st.session_state["syn_cm"]
            X_syn = st.session_state["syn_X"]
            y_syn = st.session_state["syn_y"]
            fn_syn = st.session_state["syn_feat_names"]
            mdl_syn = st.session_state["syn_model"]
            str_syn = st.session_state["syn_strengths"]
            ns_syn = st.session_state["syn_ns"]
            nv_syn = st.session_state["syn_noise_val"]

            st.markdown(
                f"#### ✅ Accuracy `{acc_syn*100:.1f}%`  ·  "
                f"Noise `{nv_syn}%`  ·  Samples `{ns_syn}`"
            )

            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
            fig_style(fig2, list(axes2))

            # Confusion matrix
            sns.heatmap(
                cm_syn,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=axes2[0],
                linewidths=0.5,
                linecolor="#1e3a5f",
            )
            axes2[0].set_title("Confusion Matrix", fontsize=11)
            axes2[0].set_xlabel("Predicted")
            axes2[0].set_ylabel("Actual")

            # Set vs learned
            learned = [
                np.exp(mdl_syn.feature_log_prob_[1, fi]) for fi in range(len(fn_syn))
            ]
            xp = np.arange(len(fn_syn))
            axes2[1].bar(
                xp - 0.2,
                str_syn,
                0.38,
                color="#0369a1",
                label="Set Strength",
                alpha=0.9,
            )
            axes2[1].bar(
                xp + 0.2,
                learned,
                0.38,
                color="#10b981",
                label="Learned P(feat|Yes)",
                alpha=0.9,
            )
            axes2[1].set_xticks(xp)
            axes2[1].set_xticklabels(fn_syn, rotation=20, ha="right", fontsize=8)
            axes2[1].set_ylim(0, 1.05)
            axes2[1].axhline(0.5, color="#475569", linestyle="--", linewidth=1)
            axes2[1].set_title("Set vs. Learned Correlation", fontsize=11)
            axes2[1].set_ylabel("Probability")
            axes2[1].legend(
                facecolor="#0d1526",
                edgecolor="#1e3a5f",
                labelcolor="#e2e8f0",
                fontsize=8,
            )
            plt.tight_layout()
            st.pyplot(fig2)

            with st.expander("👁️ First 20 generated rows"):
                prev = pd.DataFrame(X_syn[:20], columns=fn_syn)
                prev["Target"] = ["Yes" if v else "No" for v in y_syn[:20]]
                st.dataframe(prev, use_container_width=True)

            ceiling = (
                "⚠️ Accuracy approaches 50% (random guessing) — the noise ceiling!"
                if nv_syn > 35
                else "💡 Try cranking noise above 40% to watch accuracy collapse toward 50%."
            )
            st.markdown(
                f"""
<div class="card">
<b style="color:#38bdf8">🧠 Sensitivity Insight</b><br><br>
With <b>{nv_syn}% noise</b> and avg feature strength <b>{np.mean(str_syn)*100:.0f}%</b>,
the model reached <b>{acc_syn*100:.1f}% accuracy</b>.<br>{ceiling}
</div>""",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
<div class="card" style="text-align:center;padding:3rem">
<div style="font-size:3rem">🧬</div>
<h4 style="color:#38bdf8">Configure & hit ⚡ Generate</h4>
<p style="color:#64748b">Results will appear here instantly.</p>
</div>""",
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — DECISION BOUNDARY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    has_real = "model" in st.session_state
    has_syn = "syn_model" in st.session_state

    if not has_real and not has_syn:
        st.info(
            "⬅️ Train a model in **Train** or generate data in **Synthetic Lab** first."
        )
    else:
        st.markdown("### 🗺️ Decision Boundary Visualization")
        st.markdown(
            """
*Background color = what the model predicts at every coordinate in 2-D space.*  
*Dots = actual training samples. Misclassified dots sit on the "wrong" color.*
        """
        )

        sources = []
        if has_real:
            sources.append("Uploaded (real) data")
        if has_syn:
            sources.append("Synthetic Lab data")
        source = st.radio("Data source", sources, horizontal=True, key="db_src")

        if source == "Synthetic Lab data":
            X_db = st.session_state["syn_X"].astype(float)
            y_db = st.session_state["syn_y"]
            f_names = st.session_state["syn_feat_names"]
            tgt_cls = ["No", "Yes"]
        else:
            X_db = st.session_state["X"].astype(float)
            y_db = st.session_state["y"]
            f_names = st.session_state["feat_cols"]
            tgt_cls = st.session_state["target_classes"]

        n_feats = X_db.shape[1]
        proj_method = st.radio(
            "Projection",
            ["PCA — compress all features → 2D", "Manual — pick 2 features"],
            horizontal=True,
            key="db_proj",
        )

        if "Manual" in proj_method and n_feats >= 2:
            c1, c2 = st.columns(2)
            with c1:
                fx = st.selectbox("X-axis", f_names, 0, key="db_fx")
            with c2:
                fy = st.selectbox("Y-axis", f_names, min(1, n_feats - 1), key="db_fy")
            xi, yi = f_names.index(fx), f_names.index(fy)
            X2 = X_db[:, [xi, yi]]
            ax_labels = (fx, fy)
            is_pca = False
        else:
            pca = PCA(n_components=2, random_state=42)
            X2 = pca.fit_transform(X_db)
            ax_labels = (
                f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% variance)",
                f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% variance)",
            )
            is_pca = True

        # GaussianNB on 2-D data for smooth boundary
        mdl_2d = GaussianNB()
        mdl_2d.fit(X2, y_db)
        acc_2d = accuracy_score(y_db, mdl_2d.predict(X2))

        pad = 0.6
        x_min, x_max = X2[:, 0].min() - pad, X2[:, 0].max() + pad
        y_min, y_max = X2[:, 1].min() - pad, X2[:, 1].max() + pad
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 130), np.linspace(y_min, y_max, 130)
        )
        Z = mdl_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        Z_prob = mdl_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1].reshape(
            xx.shape
        )

        n_cls = len(np.unique(y_db))
        bg_colors = (
            ["#4a0a0a", "#0a3a1a"]
            if n_cls == 2
            else ["#4a0a0a", "#0a3a1a", "#0a0a4a", "#3a2a00"][:n_cls]
        )
        pt_colors = (
            ["#ef4444", "#10b981"]
            if n_cls == 2
            else ["#ef4444", "#10b981", "#6366f1", "#f59e0b"][:n_cls]
        )
        cmap_bg = ListedColormap(bg_colors[:n_cls])

        fig3, ax3 = plt.subplots(figsize=(8, 6))
        fig_style(fig3, ax3)

        # Background: probability contour fill
        if n_cls == 2:
            ax3.contourf(
                xx, yy, Z_prob, levels=20, cmap="RdYlGn", alpha=0.45, vmin=0, vmax=1
            )
        else:
            ax3.contourf(xx, yy, Z, alpha=0.5, cmap=cmap_bg)

        # Hard boundary line
        ax3.contour(xx, yy, Z, colors=["#94a3b8"], linewidths=1.5, alpha=0.8)

        # Training points
        for ci in range(n_cls):
            mask = y_db == ci
            lbl = tgt_cls[ci] if ci < len(tgt_cls) else str(ci)
            ax3.scatter(
                X2[mask, 0],
                X2[mask, 1],
                c=pt_colors[ci],
                s=60,
                edgecolors="#0d1526",
                linewidths=0.7,
                label=f"Actual: {lbl}",
                alpha=0.9,
                zorder=3,
            )

        ax3.set_xlabel(ax_labels[0], fontsize=9)
        ax3.set_ylabel(ax_labels[1], fontsize=9)
        ax3.set_title(
            f"Decision Boundary  ·  2-D accuracy {acc_2d*100:.1f}%"
            + ("\n(GaussianNB on PCA projection)" if is_pca else ""),
            fontsize=10,
        )
        ax3.legend(
            facecolor="#0d1526", edgecolor="#1e3a5f", labelcolor="#e2e8f0", fontsize=8
        )
        plt.tight_layout()
        st.pyplot(fig3)

        with st.expander("➕ Drop a new point — where does it land?"):
            pc1, pc2 = st.columns(2)
            with pc1:
                px = st.slider(
                    "X coordinate",
                    float(x_min),
                    float(x_max),
                    float((x_min + x_max) / 2),
                    key="db_px",
                )
            with pc2:
                py = st.slider(
                    "Y coordinate",
                    float(y_min),
                    float(y_max),
                    float((y_min + y_max) / 2),
                    key="db_py",
                )

            pred_pt = int(mdl_2d.predict([[px, py]])[0])
            prob_pt = mdl_2d.predict_proba([[px, py]])[0]
            lbl_pt = tgt_cls[pred_pt] if pred_pt < len(tgt_cls) else str(pred_pt)
            conf_pt = prob_pt[pred_pt]
            is_pos_p = lbl_pt.lower() in ["yes", "true", "1"]
            box_p = "pred-yes" if is_pos_p else "pred-no"
            icon_p = "✅" if is_pos_p else "❌"
            st.markdown(
                f"""
            <div class="pred-box {box_p}" style="font-size:1rem;padding:1rem">
              {icon_p} Point ({px:.2f}, {py:.2f})
              → <b>{lbl_pt}</b> &nbsp;·&nbsp; {conf_pt*100:.1f}% confidence
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
<div class="card" style="margin-top:.5rem">
<b style="color:#38bdf8">📌 Reading the map</b><br><br>
• <b>Background color intensity</b> shows confidence — deeper green/red = more certain.<br>
• <b>The boundary line</b> is where the model is exactly 50/50 — maximum uncertainty.<br>
• <b>Dots on the wrong background</b> are misclassified training samples.<br>
• {'PCA collapses all features into 2 axes — some overlap is unavoidable but the shape is real.'
   if is_pca else 'You are seeing only 2 features; the true boundary lives in higher-dimensional space.'}
</div>""",
            unsafe_allow_html=True,
        )
