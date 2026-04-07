import streamlit as st
import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB, GaussianNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import io
import warnings

warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Naive Bayes Predictor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
}

/* Dark sci-fi background */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1a2e 0%, #091322 100%);
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #38bdf8;
}

/* Title */
h1 { font-family: 'Space Mono', monospace !important; color: #38bdf8 !important; }
h2 { color: #7dd3fc !important; }
h3 { color: #93c5fd !important; }

/* Cards */
.card {
    background: rgba(14, 30, 54, 0.8);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1.2rem;
    backdrop-filter: blur(10px);
}

/* Prediction box */
.pred-box {
    text-align: center;
    padding: 2rem;
    border-radius: 16px;
    margin: 1rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    box-shadow: 0 0 40px rgba(56,189,248,0.2);
}
.pred-yes {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 2px solid #10b981;
    color: #6ee7b7;
}
.pred-no {
    background: linear-gradient(135deg, #450a0a, #7f1d1d);
    border: 2px solid #ef4444;
    color: #fca5a5;
}
.pred-neutral {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 2px solid #6366f1;
    color: #a5b4fc;
}

/* Probability bar */
.prob-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    color: #94a3b8;
    margin-bottom: 4px;
}

/* Metric cards */
.metric-card {
    background: rgba(14, 30, 54, 0.9);
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #38bdf8;
    font-weight: 700;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #38bdf8) !important;
    box-shadow: 0 0 20px rgba(56,189,248,0.4) !important;
    transform: translateY(-1px) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 2px dashed #1e3a5f;
    border-radius: 12px;
    padding: 1rem;
    background: rgba(14, 30, 54, 0.5);
}

/* Select box / radio */
.stSelectbox > div > div,
.stRadio > div {
    background: rgba(14, 30, 54, 0.8) !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}

/* Divider */
hr { border-color: #1e3a5f !important; }

/* Info / warning */
.stAlert { border-radius: 10px; }

/* Checkbox */
.stCheckbox span { color: #cbd5e1 !important; }

/* Tag pill */
.pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 99px;
    font-size: 0.72rem;
    font-family: 'Space Mono', monospace;
    margin: 2px;
}
.pill-feature { background: #0c4a6e; color: #7dd3fc; border: 1px solid #0369a1; }
.pill-target  { background: #3b0764; color: #d8b4fe; border: 1px solid #7c3aed; }
</style>
""",
    unsafe_allow_html=True,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def is_binary_column(series):
    """Check if column has Yes/No or 0/1 style values."""
    unique = set(series.dropna().astype(str).str.strip().str.lower())
    return unique.issubset({"yes", "no", "true", "false", "1", "0", "y", "n"})


def normalize_binary(series):
    mapping = {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "y": 1, "n": 0}
    return series.astype(str).str.strip().str.lower().map(mapping)


def encode_column(series):
    le = LabelEncoder()
    encoded = le.fit_transform(series.astype(str).str.strip().str.lower())
    return encoded, le


def train_model(df, feature_cols, target_col):
    X_raw = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Encode features
    encoders = {}
    X_enc = pd.DataFrame()
    all_binary = True

    for col in feature_cols:
        if is_binary_column(X_raw[col]):
            X_enc[col] = normalize_binary(X_raw[col])
        else:
            enc, le = encode_column(X_raw[col])
            X_enc[col] = enc
            encoders[col] = le
            all_binary = False

    # Encode target
    if is_binary_column(y_raw):
        y_enc = normalize_binary(y_raw)
        target_classes = ["No", "Yes"]
        target_binary = True
    else:
        y_enc, target_le = encode_column(y_raw)
        target_classes = list(target_le.classes_)
        target_binary = False
        encoders["__target__"] = target_le

    X = X_enc.values.astype(int)
    y = y_enc.values.astype(int)

    # Choose model
    if all_binary:
        model = BernoulliNB()
    else:
        model = CategoricalNB()

    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))

    return model, encoders, target_classes, target_binary, X, y, acc


def predict_sample(model, feature_cols, encoders, input_dict, target_binary):
    row = []
    for col in feature_cols:
        val = input_dict[col]
        if col in encoders:
            le = encoders[col]
            val_enc = le.transform([val.strip().lower()])[0]
            row.append(val_enc)
        else:
            mapping = {
                "yes": 1,
                "no": 0,
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0,
                "y": 1,
                "n": 0,
            }
            row.append(mapping.get(val.strip().lower(), 0))
    X_new = np.array([row])
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0]
    return pred, proba


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## NB Predictor")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown(
        """
1. **Upload** a CSV with Yes/No or categorical data  
2. **Select** your feature columns & target  
3. The model **trains** using Naive Bayes  
4. **Toggle** checkboxes to get instant predictions  
    """
    )
    st.markdown("---")
    st.markdown("### Supported Formats")
    st.markdown(
        """
- **Yes / No** columns  
- **True / False** columns  
- **1 / 0** columns  
- **Categorical** (strings)  
    """
    )
    st.markdown("---")
    st.markdown(
        """
<div style='font-family:Space Mono,monospace;font-size:0.7rem;color:#475569;text-align:center'>
Powered by sklearn · Streamlit
</div>
""",
        unsafe_allow_html=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Naive Bayes Predictor")
st.markdown(
    "*Upload a dataset, pick your columns, and get instant AI-powered predictions.*"
)
st.markdown("---")

# ── Upload ──
st.markdown("### 📂 Upload Your Dataset")
uploaded = st.file_uploader("Drop a CSV file here", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    df.columns = df.columns.str.strip()

    st.success(f"✅ Loaded **{len(df)} rows × {len(df.columns)} columns**")

    with st.expander("👁️ Preview Data", expanded=False):
        st.dataframe(df, use_container_width=True, height=200)

    st.markdown("---")
    st.markdown("### ⚙️ Configure Model")

    col1, col2 = st.columns([2, 1])

    with col1:
        all_cols = list(df.columns)
        feature_cols = st.multiselect(
            "🔷 Feature Columns (inputs)",
            options=all_cols,
            default=all_cols[:-1] if len(all_cols) > 1 else [],
            help="Select the columns that the model will use to predict.",
        )

    with col2:
        remaining = [c for c in all_cols if c not in feature_cols]
        target_col = st.selectbox(
            "🎯 Target Column (predict)",
            options=remaining if remaining else all_cols,
            index=0,
            help="The column you want to predict.",
        )

    # Show pills
    if feature_cols:
        pills_html = "".join(
            [f'<span class="pill pill-feature">{c}</span>' for c in feature_cols]
        )
        pills_html += f'<span class="pill pill-target">→ {target_col}</span>'
        st.markdown(pills_html, unsafe_allow_html=True)

    st.markdown("---")

    if feature_cols and target_col and st.button("🚀 Train Model"):
        with st.spinner("Training Naive Bayes..."):
            try:
                model, encoders, target_classes, target_binary, X, y, acc = train_model(
                    df, feature_cols, target_col
                )
                st.session_state["model"] = model
                st.session_state["encoders"] = encoders
                st.session_state["target_classes"] = target_classes
                st.session_state["target_binary"] = target_binary
                st.session_state["feature_cols"] = feature_cols
                st.session_state["target_col"] = target_col
                st.session_state["acc"] = acc
                st.session_state["X"] = X
                st.session_state["y"] = y
                st.session_state["df"] = df
                st.success("✅ Model trained successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")

    # ── Prediction Section ──
    if "model" in st.session_state:
        model = st.session_state["model"]
        encoders = st.session_state["encoders"]
        target_classes = st.session_state["target_classes"]
        target_binary = st.session_state["target_binary"]
        feat_cols = st.session_state["feature_cols"]
        tgt_col = st.session_state["target_col"]
        acc = st.session_state["acc"]
        X = st.session_state["X"]
        y = st.session_state["y"]
        df_stored = st.session_state["df"]

        st.markdown("---")
        st.markdown("### 📊 Model Performance")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{acc*100:.1f}%</div>
                <div class="metric-label">Training Accuracy</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{len(df_stored)}</div>
                <div class="metric-label">Training Samples</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with c3:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="metric-value">{len(feat_cols)}</div>
                <div class="metric-label">Features Used</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # Confusion matrix
        with st.expander("📉 Confusion Matrix & Report", expanded=False):
            y_pred_all = model.predict(X)
            fig, ax = plt.subplots(figsize=(5, 4))
            fig.patch.set_facecolor("#0d1526")
            ax.set_facecolor("#0d1526")
            cm = confusion_matrix(y, y_pred_all)
            labels = (
                target_classes
                if len(target_classes) == cm.shape[0]
                else list(range(cm.shape[0]))
            )
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
            ax.set_xlabel("Predicted", color="#94a3b8")
            ax.set_ylabel("Actual", color="#94a3b8")
            ax.tick_params(colors="#94a3b8")
            ax.set_title("Confusion Matrix", color="#7dd3fc", fontsize=13)
            plt.tight_layout()
            st.pyplot(fig)

            report = classification_report(
                y,
                y_pred_all,
                target_names=labels if len(labels) == cm.shape[0] else None,
                output_dict=True,
            )
            st.dataframe(
                pd.DataFrame(report).transpose().round(2), use_container_width=True
            )

        st.markdown("---")
        st.markdown("### 🎛️ Make a Prediction")
        st.markdown(
            "*Select values for each feature below and get an instant prediction.*"
        )

        input_dict = {}
        col_groups = [feat_cols[i : i + 3] for i in range(0, len(feat_cols), 3)]

        for group in col_groups:
            cols = st.columns(len(group))
            for i, col_name in enumerate(group):
                with cols[i]:
                    col_vals = df_stored[col_name].dropna().astype(str).str.strip()
                    unique_vals = sorted(
                        col_vals.unique().tolist(), key=lambda x: x.lower()
                    )

                    # Use radio for Yes/No, selectbox for others
                    if is_binary_column(df_stored[col_name]):
                        choice = st.radio(
                            f"**{col_name}**",
                            options=unique_vals,
                            horizontal=True,
                            key=f"input_{col_name}",
                        )
                    else:
                        choice = st.selectbox(
                            f"**{col_name}**",
                            options=unique_vals,
                            key=f"input_{col_name}",
                        )
                    input_dict[col_name] = choice

        st.markdown("")
        if st.button("🔮 Predict Now", use_container_width=True):
            try:
                pred_idx, proba = predict_sample(
                    model, feat_cols, encoders, input_dict, target_binary
                )

                # Resolve label
                if target_binary:
                    pred_label = "Yes" if pred_idx == 1 else "No"
                else:
                    le = encoders.get("__target__")
                    pred_label = (
                        le.inverse_transform([pred_idx])[0].title()
                        if le
                        else str(pred_idx)
                    )

                # Display prediction
                is_positive = str(pred_label).lower() in ["yes", "true", "1"]
                is_negative = str(pred_label).lower() in ["no", "false", "0"]

                box_class = (
                    "pred-yes"
                    if is_positive
                    else ("pred-no" if is_negative else "pred-neutral")
                )
                icon = "✅" if is_positive else ("❌" if is_negative else "🔵")

                st.markdown(
                    f"""
                <div class="pred-box {box_class}">
                    {icon} &nbsp; {tgt_col}: <span style="text-transform:uppercase">{pred_label}</span>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Probability bars
                st.markdown("#### 📈 Class Probabilities")
                for cls_idx, cls_prob in enumerate(proba):
                    if cls_idx < len(target_classes):
                        lbl = target_classes[cls_idx]
                    else:
                        lbl = str(cls_idx)
                    bar_color = (
                        "#10b981"
                        if lbl.lower() in ["yes", "true", "1"]
                        else (
                            "#ef4444"
                            if lbl.lower() in ["no", "false", "0"]
                            else "#6366f1"
                        )
                    )
                    st.markdown(
                        f'<div class="prob-label">{lbl}</div>', unsafe_allow_html=True
                    )
                    st.progress(float(cls_prob))
                    st.markdown(
                        f"<div style='text-align:right;font-family:Space Mono,monospace;"
                        f"font-size:0.8rem;color:#94a3b8;margin-top:-10px'>"
                        f"{cls_prob*100:.1f}%</div>",
                        unsafe_allow_html=True,
                    )

                # Reasoning
                with st.expander("🔍 Naive Bayes Reasoning", expanded=False):
                    st.markdown(
                        f"""
**How Naive Bayes made this decision:**

Naive Bayes calculates the probability of each class using Bayes' theorem:

> **P(class | features) ∝ P(class) × ∏ P(feature_i | class)**

For your input:
"""
                    )
                    for col_name, val in input_dict.items():
                        st.markdown(f"- **{col_name}** = `{val}`")
                    st.markdown(
                        f"""
The model multiplied the prior probability of each `{tgt_col}` class with the 
likelihood of each feature value given that class. The class with the highest 
posterior probability — **{pred_label}** ({proba[pred_idx]*100:.1f}%) — was selected as the prediction.
"""
                    )

            except Exception as e:
                st.error(f"Prediction error: {e}")

else:
    # Landing state
    st.markdown(
        """
<div class="card" style="text-align:center;padding:3rem">
    <div style="font-size:4rem">📋</div>
    <h3 style="color:#38bdf8;font-family:'Space Mono',monospace">Upload a CSV to Begin</h3>
    <p style="color:#64748b;max-width:400px;margin:auto">
        Your CSV should have feature columns (like Cough, Fever) and a target column 
        (like Flu) with Yes/No or categorical values.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Example CSV download
    example_csv = """Cough,Fever,Headache,Flu
Yes,Yes,Yes,Yes
Yes,Yes,No,Yes
Yes,No,Yes,Yes
No,Yes,Yes,Yes
Yes,No,No,No
No,Yes,No,No
No,No,Yes,No
No,No,No,No
Yes,Yes,No,No"""
    st.download_button(
        "📥 Download Example CSV",
        data=example_csv,
        file_name="flu_dataset.csv",
        mime="text/csv",
    )
