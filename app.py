
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from documents import DOCUMENTS


# ============================================================
# 1. PAGE CONFIGURATION AND THEME
# ============================================================

st.set_page_config(
    page_title="Technical AI Model Laboratory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG = "#080D18"
PANEL = "#111827"
BORDER = "#263449"
CYAN = "#22D3EE"
BLUE = "#60A5FA"
PURPLE = "#A78BFA"
GREEN = "#34D399"
ORANGE = "#FBBF24"
RED = "#FB7185"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"


st.markdown(
    f"""
    <style>
    .stApp {{
        background: {BG};
        color: {TEXT};
    }}

    [data-testid="stSidebar"] {{
        background: #0B1220;
        border-right: 1px solid {BORDER};
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1600px;
    }}

    h1, h2, h3 {{
        color: {TEXT} !important;
        letter-spacing: -0.4px;
    }}

    p, li, label {{
        color: {TEXT};
    }}

    .hero {{
        padding: 26px 30px;
        border: 1px solid {BORDER};
        border-radius: 18px;
        background:
            linear-gradient(120deg, #111827 0%, #101B30 60%,
            #17243A 100%);
        margin-bottom: 24px;
    }}

    .hero-title {{
        font-size: 31px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        font-size: 15px;
        color: {MUTED};
        line-height: 1.7;
    }}

    .eyebrow {{
        color: {CYAN};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }}

    .panel {{
        background: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
    }}

    .panel-label {{
        font-size: 12px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }}

    .panel-value {{
        font-size: 28px;
        font-weight: 800;
        color: #F8FAFC;
        margin-top: 7px;
    }}

    .panel-note {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 5px;
    }}

    .status-dot {{
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: {GREEN};
        margin-right: 8px;
    }}

    .small-muted {{
        color: {MUTED};
        font-size: 12px;
    }}

    div[data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {BORDER};
        padding: 15px;
        border-radius: 12px;
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}

    div[data-testid="stMetricValue"] {{
        color: #F8FAFC;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    .stButton > button {{
        border-radius: 9px;
        font-weight: 700;
        border: 1px solid {BORDER};
    }}

    hr {{
        border-color: {BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def panel(label, value, note=""):
    """Display a dashboard metric card."""
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-label">{label}</div>
            <div class="panel-value">{value}</div>
            <div class="panel-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_plot(fig, height=350):
    """Apply the common dark dashboard style to Plotly figures."""
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        ),
        title_font=dict(size=16, color=TEXT),
    )

    fig.update_xaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
    )
    fig.update_yaxes(
        gridcolor=BORDER,
        zerolinecolor=BORDER,
    )
    return fig


def get_document_text(doc):
    return f"{doc['title']} {doc['text']}"


# ============================================================
# 3. DOCUMENT RETRIEVAL
# ============================================================

@st.cache_resource
def build_retrieval_index():
    """Build a TF-IDF index from the synthetic document corpus."""
    corpus = [get_document_text(doc) for doc in DOCUMENTS]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(corpus)
    return vectorizer, matrix


def retrieve(query, k=3):
    """Return the top-k documents using TF-IDF cosine similarity."""
    vectorizer, matrix = build_retrieval_index()

    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).flatten()

    order = np.argsort(scores)[::-1][:k]

    results = []
    for idx in order:
        doc = DOCUMENTS[int(idx)].copy()
        doc["score"] = float(scores[idx])
        results.append(doc)

    return results


# ============================================================
# 4. NAIVE BAYES CLASSIFIER
# ============================================================

NB_TRAINING_DATA = [
    (
        "equipment temperature high cooling ventilation airflow",
        "preventive_maintenance",
    ),
    (
        "inspect equipment ventilation connectors operating logs",
        "preventive_maintenance",
    ),
    (
        "preventive inspection maintenance schedule abnormal indications",
        "preventive_maintenance",
    ),
    (
        "monitor temperature operating limits cooling system",
        "preventive_maintenance",
    ),
    (
        "communication intermittent signal degradation interference",
        "troubleshooting",
    ),
    (
        "packet loss link quality congestion communication",
        "troubleshooting",
    ),
    (
        "communication dropping connector problems signal quality",
        "troubleshooting",
    ),
    (
        "investigate link degradation and record packet loss",
        "troubleshooting",
    ),
    (
        "unexpected restarts watchdog events software faults",
        "fault_isolation",
    ),
    (
        "unstable power causes resets intermittent operation",
        "fault_isolation",
    ),
    (
        "review restart event logs and power observations",
        "fault_isolation",
    ),
    (
        "fault isolation diagnose reset power stability",
        "fault_isolation",
    ),
]


@st.cache_resource
def build_nb_model():
    texts = [item[0] for item in NB_TRAINING_DATA]
    labels = [item[1] for item in NB_TRAINING_DATA]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
            ),
        ),
        ("classifier", MultinomialNB(alpha=0.5)),
    ])

    model.fit(texts, labels)
    return model


def predict_category(query):
    model = build_nb_model()
    predicted = model.predict([query])[0]

    probabilities = model.predict_proba([query])[0]
    classes = model.named_steps["classifier"].classes_

    probability_map = {
        str(category): float(probability)
        for category, probability in zip(classes, probabilities)
    }

    return predicted, probability_map


def category_rank(query, predicted_category, k=3, bonus=0.15):
    """
    Category-aware retrieval.

    The category prediction is supplied by the Naive Bayes classifier.
    A bonus is added to documents in that category.
    """
    results = retrieve(query, k=len(DOCUMENTS))

    for doc in results:
        doc["adjusted_score"] = (
            doc["score"]
            + (
                bonus
                if doc["category"] == predicted_category
                else 0.0
            )
        )

    results.sort(
        key=lambda doc: doc["adjusted_score"],
        reverse=True,
    )

    return results[:k]


# ============================================================
# 5. HIDDEN MARKOV MODEL AND VITERBI
# ============================================================

HMM_STATES = ["Normal", "Degraded", "Fault"]

HMM_OBSERVATIONS = [
    "Normal temperature",
    "Elevated temperature",
    "High packet loss",
    "Repeated restart",
]

HMM_TRANSITION = np.array([
    [0.80, 0.17, 0.03],
    [0.15, 0.65, 0.20],
    [0.05, 0.15, 0.80],
])

HMM_EMISSION = np.array([
    [0.75, 0.10, 0.12, 0.03],
    [0.15, 0.45, 0.32, 0.08],
    [0.05, 0.10, 0.15, 0.70],
])

HMM_INITIAL = np.array([0.80, 0.15, 0.05])


def viterbi(observation_sequence):
    """Infer the most likely hidden-state sequence."""
    obs_indices = [
        HMM_OBSERVATIONS.index(obs)
        for obs in observation_sequence
    ]

    n_states = len(HMM_STATES)
    n_steps = len(obs_indices)

    log_transition = np.log(HMM_TRANSITION + 1e-12)
    log_emission = np.log(HMM_EMISSION + 1e-12)
    log_initial = np.log(HMM_INITIAL + 1e-12)

    dp = np.zeros((n_steps, n_states))
    backpointer = np.zeros(
        (n_steps, n_states),
        dtype=int,
    )

    first_obs = obs_indices[0]
    dp[0] = (
        log_initial
        + log_emission[:, first_obs]
    )

    for t in range(1, n_steps):
        obs = obs_indices[t]

        for state in range(n_states):
            candidates = (
                dp[t - 1]
                + log_transition[:, state]
            )

            best_previous = int(np.argmax(candidates))

            dp[t, state] = (
                candidates[best_previous]
                + log_emission[state, obs]
            )

            backpointer[t, state] = best_previous

    last_state = int(np.argmax(dp[-1]))
    best_path = [last_state]

    for t in range(n_steps - 1, 0, -1):
        last_state = backpointer[t, last_state]
        best_path.append(int(last_state))

    best_path.reverse()

    state_sequence = [
        HMM_STATES[state]
        for state in best_path
    ]

    final_probs = np.exp(dp[-1] - np.max(dp[-1]))
    final_probs = final_probs / final_probs.sum()

    state_probabilities = {
        HMM_STATES[i]: float(final_probs[i])
        for i in range(n_states)
    }

    return state_sequence, state_probabilities


# ============================================================
# 6. VARIATIONAL AUTOENCODER
# ============================================================

FEATURES = [
    "Temperature",
    "Vibration",
    "Voltage",
    "Packet Loss",
]


class TelemetryVAE(nn.Module):
    def __init__(self, input_dim=4, latent_dim=2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 12),
            nn.ReLU(),
        )

        self.mu = nn.Linear(12, latent_dim)
        self.logvar = nn.Linear(12, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 12),
            nn.ReLU(),
            nn.Linear(12, input_dim),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        return self.mu(hidden), self.logvar(hidden)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


@st.cache_resource
def train_vae():
    """Train a small VAE using synthetic normal telemetry."""
    np.random.seed(42)
    torch.manual_seed(42)

    normal_data = np.column_stack([
        np.random.normal(55, 5, 900),
        np.random.normal(1.0, 0.22, 900),
        np.random.normal(28, 0.8, 900),
        np.clip(
            np.random.normal(0.6, 0.25, 900),
            0,
            None,
        ),
    ])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(normal_data)

    tensor_data = torch.tensor(
        scaled_data,
        dtype=torch.float32,
    )

    dataset = TensorDataset(tensor_data)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
    )

    model = TelemetryVAE(
        input_dim=4,
        latent_dim=2,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
    )

    epochs = 120

    model.train()

    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()

            reconstruction, mu, logvar = model(batch)

            reconstruction_loss = nn.functional.mse_loss(
                reconstruction,
                batch,
                reduction="mean",
            )

            kl_loss = -0.5 * torch.mean(
                1
                + logvar
                - mu.pow(2)
                - logvar.exp()
            )

            loss = reconstruction_loss + 0.01 * kl_loss

            loss.backward()
            optimizer.step()

    model.eval()

    with torch.no_grad():
        reconstruction, _, _ = model(tensor_data)

        train_errors = torch.mean(
            (tensor_data - reconstruction) ** 2,
            dim=1,
        ).numpy()

    threshold = float(np.percentile(train_errors, 95))

    return model, scaler, normal_data, threshold, train_errors


def analyze_telemetry(values):
    model, scaler, _, threshold, _ = train_vae()

    scaled = scaler.transform(
        np.array(values, dtype=float).reshape(1, -1)
    )

    tensor = torch.tensor(
        scaled,
        dtype=torch.float32,
    )

    with torch.no_grad():
        reconstruction, mu, logvar = model(tensor)

    reconstructed_scaled = reconstruction.numpy()
    reconstructed_original = scaler.inverse_transform(
        reconstructed_scaled
    )[0]

    error = float(
        np.mean(
            (scaled - reconstructed_scaled) ** 2
        )
    )

    latent = mu.numpy()[0]

    return {
        "error": error,
        "threshold": threshold,
        "anomaly": error > threshold,
        "reconstruction": reconstructed_original,
        "latent": latent,
    }


# ============================================================
# 7. GROUND-TRUTH RAG EVALUATION
# ============================================================

EVALUATION_SET = [
    {
        "question": "Why is the equipment temperature increasing?",
        "relevant": ["MAN-001"],
    },
    {
        "question": "Communication keeps dropping intermittently.",
        "relevant": ["MAN-002"],
    },
    {
        "question": "The equipment is restarting unexpectedly.",
        "relevant": ["MAN-003", "MAN-006"],
    },
    {
        "question": "How should increasing packet loss be investigated?",
        "relevant": ["MAN-004"],
    },
    {
        "question": "What should a preventive inspection include?",
        "relevant": ["MAN-005"],
    },
    {
        "question": "What could cause resets due to unstable power?",
        "relevant": ["MAN-006", "MAN-003"],
    },
]


def evaluate_retrieval(ranked_docs, relevant_ids, k=3):
    """Return Recall@k and reciprocal rank."""
    relevant = set(relevant_ids)

    ranked_ids = [
        doc["id"]
        for doc in ranked_docs
    ]

    top_k = ranked_ids[:k]

    recall = (
        len(set(top_k) & relevant) / len(relevant)
        if relevant
        else 0.0
    )

    reciprocal_rank = 0.0

    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            reciprocal_rank = 1.0 / rank
            break

    return recall, reciprocal_rank


def run_evaluation():
    metric_rows = []
    detail_rows = []

    for item in EVALUATION_SET:
        query = item["question"]
        relevant_ids = item["relevant"]

        baseline = retrieve(query, k=3)

        predicted_category, _ = predict_category(query)

        category_results = category_rank(
            query,
            predicted_category,
            k=3,
        )

        baseline_recall, baseline_rr = evaluate_retrieval(
            baseline,
            relevant_ids,
        )

        category_recall, category_rr = evaluate_retrieval(
            category_results,
            relevant_ids,
        )

        baseline_ids = [
            doc["id"] for doc in baseline
        ]

        category_ids = [
            doc["id"] for doc in category_results
        ]

        metric_rows.append({
            "Question": query,
            "Baseline Recall@3": baseline_recall,
            "Category Recall@3": category_recall,
            "Baseline RR": baseline_rr,
            "Category RR": category_rr,
        })

        detail_rows.append({
            "Question": query,
            "Relevant IDs": ", ".join(relevant_ids),
            "NB predicted category": predicted_category,
            "Baseline ranking": " → ".join(baseline_ids),
            "Category-aware ranking": " → ".join(category_ids),
            "Ranking changed": baseline_ids != category_ids,
            "Baseline hit": bool(
                set(baseline_ids) & set(relevant_ids)
            ),
            "Category-aware hit": bool(
                set(category_ids) & set(relevant_ids)
            ),
        })

    metrics_df = pd.DataFrame(metric_rows)
    details_df = pd.DataFrame(detail_rows)

    summary = {
        "baseline_recall": float(
            metrics_df["Baseline Recall@3"].mean()
        ),
        "category_recall": float(
            metrics_df["Category Recall@3"].mean()
        ),
        "baseline_mrr": float(
            metrics_df["Baseline RR"].mean()
        ),
        "category_mrr": float(
            metrics_df["Category RR"].mean()
        ),
    }

    return metrics_df, details_df, summary


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style="padding: 10px 4px 22px 4px;">
            <div style="font-size: 28px; font-weight: 900;
                        color: {CYAN};">
                MM AI
            </div>
            <div style="font-size: 12px; color: {MUTED};
                        letter-spacing: 1px;">
                MODEL LABORATORY
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Select laboratory",
        [
            "VAE Laboratory",
            "Naive Bayes Laboratory",
            "HMM Laboratory",
            "Agentic RAG",
            "RAG Evaluation Lab",
        ],
        label_visibility="collapsed",
    )

# ============================================================
# M² AI — SHARED BRAND HEADER
# ============================================================

import os

LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "m2_ai_logo.png",
)

header_left, header_right = st.columns([1, 5])

with header_left:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=100)
    else:
        st.markdown("## M² AI")

with header_right:
    st.markdown(
        """
        <div style="
            padding: 8px 0 16px 0;
            border-bottom: 1px solid #263449;
            margin-bottom: 18px;
        ">
            <div style="
                color: #22D3EE;
                font-size: 11px;
                font-weight: 800;
                letter-spacing: 2px;
            ">
                M² AI · MODEL LABORATORY
            </div>
            <div style="
                color: #F8FAFC;
                font-size: 27px;
                font-weight: 800;
            ">
                Technical AI Model Laboratory
            </div>
            <div style="
                color: #94A3B8;
                font-size: 13px;
            ">
                Explore · Experiment · Evaluate
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.markdown(
        f"""
        <div style="color: {GREEN}; font-size: 13px;">
            <span class="status-dot"></span>Application loaded
        </div>
        <div style="color: {MUTED}; font-size: 12px;
                    margin-top: 8px;">
            Synthetic demonstration data<br>
            Local model inference
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Technical AI Model Laboratory")
    st.caption("Educational prototype • Synthetic data only")


# ============================================================
# 9. COMMON PAGE HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">INTELLIGENT SYSTEMS • EXPERIMENTAL LAB</div>
        <div class="hero-title">Technical AI Model Laboratory</div>
        <div class="hero-subtitle">
            Explore anomaly detection, probabilistic classification,
            hidden-state inference, and retrieval-augmented generation
            through interactive experiments.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 10. VAE LABORATORY
# ============================================================

if page == "VAE Laboratory":

    st.markdown("## VAE Laboratory")
    st.caption(
        "Variational Autoencoder • Telemetry reconstruction • "
        "Anomaly detection"
    )

    model, scaler, normal_data, threshold, train_errors = train_vae()

    st.markdown("### Live telemetry input")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        temperature = st.slider(
            "Temperature",
            min_value=20.0,
            max_value=120.0,
            value=55.0,
            step=1.0,
            help="Synthetic equipment temperature.",
        )

    with c2:
        vibration = st.slider(
            "Vibration",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
        )

    with c3:
        voltage = st.slider(
            "Voltage",
            min_value=15.0,
            max_value=40.0,
            value=28.0,
            step=0.5,
        )

    with c4:
        packet_loss = st.slider(
            "Packet loss",
            min_value=0.0,
            max_value=10.0,
            value=0.6,
            step=0.1,
        )

    telemetry = [
        temperature,
        vibration,
        voltage,
        packet_loss,
    ]

    analysis = analyze_telemetry(telemetry)

    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        panel(
            "Reconstruction error",
            f"{analysis['error']:.4f}",
            "Mean squared error in standardized feature space",
        )

    with metric2:
        panel(
            "Training threshold",
            f"{analysis['threshold']:.4f}",
            "95th percentile of synthetic training errors",
        )

    with metric3:
        status = (
            "ANOMALY DETECTED"
            if analysis["anomaly"]
            else "NORMAL RANGE"
        )

        panel(
            "Model assessment",
            status,
            "Illustrative model output; not a safety diagnosis",
        )

    if analysis["anomaly"]:
        st.warning(
            "The input exceeds the model's illustrative "
            "reconstruction-error threshold."
        )
    else:
        st.success(
            "The input is within the model's learned "
            "reconstruction-error threshold."
        )

    st.markdown("### Input versus reconstruction")

    reconstruction_df = pd.DataFrame({
        "Feature": FEATURES,
        "Input": telemetry,
        "Reconstruction": analysis["reconstruction"],
    })

    reconstruction_long = reconstruction_df.melt(
        id_vars="Feature",
        var_name="Measurement",
        value_name="Value",
    )

    fig = px.bar(
        reconstruction_long,
        x="Feature",
        y="Value",
        color="Measurement",
        barmode="group",
        title="Input telemetry vs VAE reconstruction",
        color_discrete_sequence=[CYAN, PURPLE],
    )

    st.plotly_chart(
        style_plot(fig, 360),
        use_container_width=True,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("### Latent representation")

        latent_df = pd.DataFrame({
            "Latent dimension": ["z1", "z2"],
            "Value": analysis["latent"],
        })

        fig = px.bar(
            latent_df,
            x="Latent dimension",
            y="Value",
            title="Encoder latent mean",
            color="Latent dimension",
            color_discrete_sequence=[CYAN, PURPLE],
        )

        st.plotly_chart(
            style_plot(fig, 300),
            use_container_width=True,
        )

    with right:
        st.markdown("### Training reconstruction errors")

        fig = px.histogram(
            x=train_errors,
            nbins=35,
            title="Synthetic normal training distribution",
            labels={"x": "Reconstruction error"},
            color_discrete_sequence=[BLUE],
        )

        fig.add_vline(
            x=threshold,
            line_dash="dash",
            line_color=ORANGE,
            annotation_text="Threshold",
        )

        fig.add_vline(
            x=analysis["error"],
            line_dash="solid",
            line_color=RED,
            annotation_text="Current input",
        )

        st.plotly_chart(
            style_plot(fig, 300),
            use_container_width=True,
        )

    st.markdown("### Telemetry summary")

    st.dataframe(
        reconstruction_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Teaching point: a VAE learns a compressed latent representation "
        "and reconstructs inputs. A high reconstruction error can flag "
        "unusual inputs, but this synthetic demonstration is not a "
        "validated anomaly detector."
    )


# ============================================================
# 11. NAIVE BAYES LABORATORY
# ============================================================

elif page == "Naive Bayes Laboratory":

    st.markdown("## Naive Bayes Laboratory")
    st.caption(
        "Text classification • Probabilities • Category prediction"
    )

    model = build_nb_model()

    query = st.text_area(
        "Enter a technical symptom or maintenance query",
        value="The equipment is restarting unexpectedly due to unstable power.",
        height=100,
    )

    if st.button("Classify query", type="primary"):
        predicted, probabilities = predict_category(query)

        st.markdown("### Prediction")

        c1, c2 = st.columns(2)

        with c1:
            panel(
                "Predicted category",
                predicted.replace("_", " ").title(),
                "Multinomial Naive Bayes",
            )

        with c2:
            confidence = probabilities[predicted]

            panel(
                "Model probability",
                f"{confidence:.1%}",
                "Probability within this model's class distribution",
            )

        st.markdown("### Category probabilities")

        prob_df = pd.DataFrame({
            "Category": list(probabilities.keys()),
            "Probability": list(probabilities.values()),
        }).sort_values("Probability", ascending=True)

        fig = px.bar(
            prob_df,
            x="Probability",
            y="Category",
            orientation="h",
            range_x=[0, 1],
            text=prob_df["Probability"].map(
                lambda x: f"{x:.1%}"
            ),
            title="Predicted class probabilities",
            color="Probability",
            color_continuous_scale="Blues",
        )

        st.plotly_chart(
            style_plot(fig, 320),
            use_container_width=True,
        )

        st.markdown("### Retrieved documents")

        results = category_rank(
            query,
            predicted,
            k=3,
        )

        for rank, doc in enumerate(results, start=1):
            with st.container(border=True):
                st.markdown(
                    f"**#{rank} — {doc['title']}**"
                )
                st.caption(
                    f"{doc['id']} • {doc['category']}"
                )
                st.write(doc["text"])
                st.write(
                    f"Similarity: `{doc['score']:.3f}`  "
                    f"| Adjusted score: "
                    f"`{doc['adjusted_score']:.3f}`"
                )

    else:
        st.markdown("### Model training data")

        training_df = pd.DataFrame(
            NB_TRAINING_DATA,
            columns=["Example text", "Category"],
        )

        st.dataframe(
            training_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            "Enter a query and click Classify query to see how the "
            "model predicts a category and how that category can "
            "influence retrieval."
        )

    st.markdown("### How Naive Bayes works")

    st.write(
        "Naive Bayes estimates the probability of a class given the "
        "observed words. It applies Bayes' theorem and assumes that "
        "features are conditionally independent given the class."
    )

    st.latex(
        r"P(C\mid x_1,\ldots,x_n)"
        r"\propto P(C)\prod_{i=1}^{n}P(x_i\mid C)"
    )

    st.caption(
        "This model uses a small, manually authored synthetic training "
        "set. Its probabilities are not calibrated operational confidence."
    )


# ============================================================
# 12. HMM LABORATORY
# ============================================================

elif page == "HMM Laboratory":

    st.markdown("## Hidden Markov Model Laboratory")
    st.caption(
        "Hidden states • Observation sequences • Viterbi decoding"
    )

    st.write(
        "Select a sequence of observable symptoms. The HMM estimates "
        "the most likely sequence of hidden equipment states."
    )

    observation_options = HMM_OBSERVATIONS

    default_sequence = [
        "Normal temperature",
        "Elevated temperature",
        "High packet loss",
        "Repeated restart",
    ]

    selected_sequence = st.multiselect(
        "Observation sequence, in time order",
        options=observation_options,
        default=default_sequence,
    )

    if not selected_sequence:
        st.warning("Select at least one observation to run Viterbi.")
    else:
        states, final_probs = viterbi(selected_sequence)

        st.markdown("### Inferred state sequence")

        state_df = pd.DataFrame({
            "Time step": np.arange(1, len(states) + 1),
            "Observation": selected_sequence,
            "Inferred state": states,
        })

        st.dataframe(
            state_df,
            use_container_width=True,
            hide_index=True,
        )

        state_colors = {
            "Normal": GREEN,
            "Degraded": ORANGE,
            "Fault": RED,
        }

        fig = go.Figure()

        for state_name in HMM_STATES:
            mask = [
                state == state_name
                for state in states
            ]

            times = [
                i + 1
                for i, present in enumerate(mask)
                if present
            ]

            if times:
                fig.add_trace(
                    go.Scatter(
                        x=times,
                        y=[state_name] * len(times),
                        mode="markers+text",
                        text=[state_name] * len(times),
                        textposition="top center",
                        marker=dict(
                            size=17,
                            color=state_colors[state_name],
                        ),
                        name=state_name,
                    )
                )

        fig.update_layout(
            title="Most likely hidden state over time",
            xaxis_title="Observation time step",
            yaxis_title="Inferred state",
        )

        st.plotly_chart(
            style_plot(fig, 330),
            use_container_width=True,
        )

        st.markdown("### Final-step state probabilities")

        prob_df = pd.DataFrame({
            "State": list(final_probs.keys()),
            "Probability": list(final_probs.values()),
        })

        fig = px.bar(
            prob_df,
            x="State",
            y="Probability",
            range_y=[0, 1],
            text=prob_df["Probability"].map(
                lambda x: f"{x:.1%}"
            ),
            color="State",
            color_discrete_map={
                "Normal": GREEN,
                "Degraded": ORANGE,
                "Fault": RED,
            },
            title="Normalized final-state scores",
        )

        st.plotly_chart(
            style_plot(fig, 300),
            use_container_width=True,
        )

        st.markdown("### HMM transition matrix")

        transition_df = pd.DataFrame(
            HMM_TRANSITION,
            index=HMM_STATES,
            columns=HMM_STATES,
        )

        st.dataframe(
            transition_df.style.format("{:.2f}"),
            use_container_width=True,
        )

        st.markdown("### HMM emission matrix")

        emission_df = pd.DataFrame(
            HMM_EMISSION,
            index=HMM_STATES,
            columns=HMM_OBSERVATIONS,
        )

        st.dataframe(
            emission_df.style.format("{:.2f}"),
            use_container_width=True,
        )

    st.info(
        "Teaching point: the Viterbi algorithm finds the most likely "
        "hidden-state sequence given the observations and the model "
        "parameters. These transition and emission probabilities are "
        "illustrative, not learned from real equipment histories."
    )


# ============================================================
# 13. AGENTIC RAG
# ============================================================

elif page == "Agentic RAG":

    st.markdown("## Agentic RAG")
    st.caption(
        "Query understanding • Retrieval • Category-aware reranking"
    )

    query = st.text_area(
        "Ask a technical question",
        value="The equipment is restarting unexpectedly. What should I investigate?",
        height=100,
    )

    k = st.slider(
        "Number of documents to retrieve",
        min_value=1,
        max_value=min(6, len(DOCUMENTS)),
        value=min(3, len(DOCUMENTS)),
    )

    method = st.radio(
        "Retrieval strategy",
        [
            "Baseline TF-IDF",
            "Naive Bayes category-aware",
        ],
        horizontal=True,
    )

    if st.button("Run retrieval", type="primary"):
        predicted_category, probabilities = predict_category(query)

        st.markdown("### Query analysis")

        st.write(
            f"**Predicted category:** "
            f"`{predicted_category}`"
        )

        if method == "Baseline TF-IDF":
            results = retrieve(query, k=k)
        else:
            results = category_rank(
                query,
                predicted_category,
                k=k,
            )

        st.markdown("### Retrieved evidence")

        for rank, doc in enumerate(results, start=1):
            with st.container(border=True):
                top_col, score_col = st.columns([4, 1])

                with top_col:
                    st.markdown(
                        f"### {rank}. {doc['title']}"
                    )
                    st.caption(
                        f"Source: {doc['id']}  •  "
                        f"Category: {doc['category']}"
                    )

                with score_col:
                    score = doc.get(
                        "adjusted_score",
                        doc["score"],
                    )
                    st.metric("Score", f"{score:.3f}")

                st.write(doc["text"])

        with st.expander("View classifier probabilities"):
            probability_df = pd.DataFrame({
                "Category": list(probabilities.keys()),
                "Probability": list(probabilities.values()),
            }).sort_values("Probability", ascending=False)

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("### Retrieval score comparison")

        if method == "Naive Bayes category-aware":
            score_key = "adjusted_score"
            chart_title = "Category-adjusted retrieval scores"
        else:
            score_key = "score"
            chart_title = "Baseline TF-IDF similarity scores"

        score_df = pd.DataFrame([
            {
                "Document": doc["id"],
                "Title": doc["title"],
                "Score": doc[score_key],
            }
            for doc in results
        ])

        fig = px.bar(
            score_df,
            x="Document",
            y="Score",
            hover_name="Title",
            title=chart_title,
            color="Score",
            color_continuous_scale="Teal",
        )

        st.plotly_chart(
            style_plot(fig, 320),
            use_container_width=True,
        )

    st.markdown("### Current prototype architecture")

    st.graphviz_chart(
        """
        digraph RAG {
            rankdir=LR;
            bgcolor="transparent";
            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#111827",
                color="#22D3EE",
                fontcolor="white",
                fontname="Arial"
            ];
            edge [
                color="#64748B",
                fontcolor="white",
                fontname="Arial"
            ];

            Q [label="User Query"];
            C [label="Naive Bayes\\nCategory"];
            R [label="TF-IDF\\nRetrieval"];
            RR [label="Category-aware\\nReranking"];
            D [label="Retrieved\\nDocuments"];

            Q -> R;
            Q -> C;
            R -> RR;
            C -> RR;
            RR -> D;
        }
        """
    )

    st.info(
        "This is a retrieval prototype, not a complete autonomous "
        "agent or a production generative-answer system. It retrieves "
        "evidence from synthetic documents but does not call an LLM "
        "to generate a grounded answer."
    )


# ============================================================
# 14. RAG EVALUATION LAB
# ============================================================

elif page == "RAG Evaluation Lab":

    st.markdown("## RAG Evaluation Laboratory")
    st.caption(
        "Ground-truth evaluation • Recall@3 • MRR • "
        "Ranking changes and failure analysis"
    )

    st.markdown(
        """
        This experiment compares two retrieval strategies over a fixed
        set of labelled synthetic technical questions:

        - **Baseline RAG:** TF-IDF similarity only.
        - **Category-aware RAG:** Naive Bayes predicts a category, and
          matching documents receive a ranking bonus.
        """
    )

    st.warning(
        "The relevance labels are manually assigned teaching labels. "
        "Review them if a question has more than one valid answer."
    )

    # Evaluation controls
    bonus = st.slider(
        "Category ranking bonus",
        min_value=0.0,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help=(
            "This value is added to the TF-IDF similarity score "
            "for documents in the predicted category."
        ),
    )

    def evaluate_with_bonus(bonus_value):
        metric_rows = []
        detail_rows = []

        for item in EVALUATION_SET:
            query = item["question"]
            relevant_ids = item["relevant"]

            baseline = retrieve(query, k=3)

            predicted_category, probabilities = predict_category(
                query
            )

            category_results = category_rank(
                query,
                predicted_category,
                k=3,
                bonus=bonus_value,
            )

            base_recall, base_rr = evaluate_retrieval(
                baseline,
                relevant_ids,
            )

            cat_recall, cat_rr = evaluate_retrieval(
                category_results,
                relevant_ids,
            )

            baseline_ids = [
                doc["id"] for doc in baseline
            ]

            category_ids = [
                doc["id"] for doc in category_results
            ]

            metric_rows.append({
                "Question": query,
                "Baseline Recall@3": base_recall,
                "Category Recall@3": cat_recall,
                "Baseline RR": base_rr,
                "Category RR": cat_rr,
            })

            detail_rows.append({
                "Question": query,
                "Relevant IDs": ", ".join(relevant_ids),
                "NB predicted category": predicted_category,
                "Baseline ranking": " → ".join(baseline_ids),
                "Category-aware ranking": " → ".join(category_ids),
                "Ranking changed": baseline_ids != category_ids,
                "Baseline hit": bool(
                    set(baseline_ids) & set(relevant_ids)
                ),
                "Category-aware hit": bool(
                    set(category_ids) & set(relevant_ids)
                ),
            })

        return (
            pd.DataFrame(metric_rows),
            pd.DataFrame(detail_rows),
        )

    metrics_df, details_df = evaluate_with_bonus(bonus)

    # Overall metrics
    baseline_recall = metrics_df[
        "Baseline Recall@3"
    ].mean()

    category_recall = metrics_df[
        "Category Recall@3"
    ].mean()

    baseline_mrr = metrics_df["Baseline RR"].mean()
    category_mrr = metrics_df["Category RR"].mean()

    st.markdown("### Overall performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Baseline Recall@3",
        f"{baseline_recall:.1%}",
    )

    c2.metric(
        "Category Recall@3",
        f"{category_recall:.1%}",
        delta=f"{category_recall - baseline_recall:+.1%}",
    )

    c3.metric(
        "Baseline MRR",
        f"{baseline_mrr:.3f}",
    )

    c4.metric(
        "Category MRR",
        f"{category_mrr:.3f}",
        delta=f"{category_mrr - baseline_mrr:+.3f}",
    )

    st.markdown("### Metric comparison")

    chart_df = pd.DataFrame({
        "Metric": ["Recall@3", "MRR"],
        "Baseline RAG": [
            baseline_recall,
            baseline_mrr,
        ],
        "Category-aware RAG": [
            category_recall,
            category_mrr,
        ],
    }).melt(
        id_vars="Metric",
        var_name="System",
        value_name="Score",
    )

    fig = px.bar(
        chart_df,
        x="Metric",
        y="Score",
        color="System",
        barmode="group",
        range_y=[0, 1],
        text_auto=".2f",
        title="Baseline vs Category-aware Retrieval",
        color_discrete_sequence=[BLUE, CYAN],
    )

    st.plotly_chart(
        style_plot(fig, 370),
        use_container_width=True,
    )

    st.markdown("### Question-level performance")

    question_chart_df = metrics_df.melt(
        id_vars="Question",
        value_vars=[
            "Baseline Recall@3",
            "Category Recall@3",
        ],
        var_name="System",
        value_name="Recall",
    )

    fig = px.bar(
        question_chart_df,
        x="Recall",
        y="Question",
        color="System",
        barmode="group",
        orientation="h",
        range_x=[0, 1],
        title="Recall@3 for each evaluation question",
        color_discrete_sequence=[BLUE, CYAN],
    )

    st.plotly_chart(
        style_plot(fig, 470),
        use_container_width=True,
    )

    st.markdown("### Ranking changes and failure analysis")

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander("View question-level metric details"):
        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True,
        )

    changed_count = int(
        details_df["Ranking changed"].sum()
    )

    improved_count = int(
        (
            details_df["Category-aware hit"]
            & ~details_df["Baseline hit"]
        ).sum()
    )

    regressed_count = int(
        (
            details_df["Baseline hit"]
            & ~details_df["Category-aware hit"]
        ).sum()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rankings changed",
        f"{changed_count}/{len(EVALUATION_SET)}",
    )

    c2.metric(
        "New successful hits",
        improved_count,
    )

    c3.metric(
        "Lost successful hits",
        regressed_count,
    )

    st.markdown("### Inspect one evaluation question")

    selected_question = st.selectbox(
        "Choose a question",
        options=[
            item["question"]
            for item in EVALUATION_SET
        ],
    )

    selected_item = next(
        item
        for item in EVALUATION_SET
        if item["question"] == selected_question
    )

    selected_relevant = selected_item["relevant"]

    baseline_selected = retrieve(
        selected_question,
        k=3,
    )

    selected_category, selected_probs = predict_category(
        selected_question
    )

    category_selected = category_rank(
        selected_question,
        selected_category,
        k=3,
        bonus=bonus,
    )

    left, right = st.columns(2)

    with left:
        st.markdown("#### Baseline results")

        for rank, doc in enumerate(
            baseline_selected,
            start=1,
        ):
            is_relevant = doc["id"] in selected_relevant

            if is_relevant:
                st.success(
                    f"#{rank} {doc['id']} — {doc['title']} "
                    f"(relevant)"
                )
            else:
                st.write(
                    f"#{rank} {doc['id']} — {doc['title']}"
                )

            st.caption(
                f"TF-IDF score: {doc['score']:.3f}"
            )

    with right:
        st.markdown("#### Category-aware results")

        st.caption(
            f"Naive Bayes category: {selected_category}"
        )

        for rank, doc in enumerate(
            category_selected,
            start=1,
        ):
            is_relevant = doc["id"] in selected_relevant

            if is_relevant:
                st.success(
                    f"#{rank} {doc['id']} — {doc['title']} "
                    f"(relevant)"
                )
            else:
                st.write(
                    f"#{rank} {doc['id']} — {doc['title']}"
                )

            st.caption(
                f"Similarity: {doc['score']:.3f} • "
                f"Adjusted: {doc['adjusted_score']:.3f}"
            )

    st.markdown("### Ground-truth labels")

    ground_truth_df = pd.DataFrame([
        {
            "Question": item["question"],
            "Relevant document IDs": ", ".join(
                item["relevant"]
            ),
        }
        for item in EVALUATION_SET
    ])

    st.dataframe(
        ground_truth_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        "Interpretation: Recall@3 measures the fraction of labelled "
        "relevant documents found in the top three. MRR rewards finding "
        "the first relevant result near the top. A ranking change is not "
        "automatically an improvement; use the relevance labels to judge it."
    )

    st.caption(
        "Scope: six synthetic documents and six manually labelled "
        "questions. This is an educational experiment, not a statistically "
        "robust benchmark. Results depend on the labels, training examples, "
        "similarity scores, and category bonus."
    )


# ============================================================
# 15. FOOTER
# ============================================================

st.divider()

st.markdown(
    f"""
    <div style="text-align: center; padding: 12px 0 5px 0;">
        <div style="font-size: 12px; color: {MUTED};">
            TECHNICAL AI MODEL LABORATORY
        </div>
        <div style="font-size: 11px; color: #64748B;
                    margin-top: 5px;">
            VAE • Naive Bayes • HMM • RAG • Evaluation
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
