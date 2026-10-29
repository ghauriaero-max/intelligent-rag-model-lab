import os

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

BG = "#F4F7FB"
PANEL = "#FFFFFF"
BORDER = "#D5DEEA"
CYAN = "#0891B2"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
GREEN = "#059669"
ORANGE = "#D97706"
RED = "#E11D48"
TEXT = "#0F172A"
MUTED = "#64748B"


st.markdown(
    f"""
    <style>
    /* ---------- Base ---------- */
    html, body, .stApp {{
        font-family: -apple-system, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(900px 500px at 8% -10%,
                rgba(8, 145, 178, 0.10), transparent 60%),
            radial-gradient(800px 500px at 100% 0%,
                rgba(124, 58, 237, 0.08), transparent 55%),
            {BG};
        color: {TEXT};
    }}

    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ background: transparent; }}

    .block-container {{
        padding-top: 1.6rem;
        padding-bottom: 2.5rem;
        max-width: 1600px;
    }}

    /* ---------- Typography ---------- */
    h1, h2, h3, h4 {{
        color: {TEXT} !important;
        letter-spacing: -0.3px;
    }}

    h2 {{
        font-weight: 800 !important;
        padding-bottom: 10px;
        margin-bottom: 4px;
        border-bottom: 1px solid {BORDER};
        background: linear-gradient(90deg, {CYAN}, {PURPLE}) left bottom
                    / 64px 2px no-repeat;
    }}

    h3 {{
        font-weight: 700 !important;
        font-size: 1.15rem !important;
        margin-top: 1.4rem !important;
    }}

    h4 {{
        font-weight: 700 !important;
        font-size: 1rem !important;
        color: #334155 !important;
    }}

    p, li, label {{ color: {TEXT}; }}

    [data-testid="stCaptionContainer"] {{ color: {MUTED}; }}

    /* ---------- Hero ---------- */
    .hero {{
        position: relative;
        padding: 22px 28px;
        border: 1px solid {BORDER};
        border-radius: 18px;
        background:
            linear-gradient(120deg, rgba(8, 145, 178, 0.08),
                            rgba(124, 58, 237, 0.07)),
            #FFFFFF;
        box-shadow: 0 8px 26px rgba(15, 23, 42, 0.08);
        margin-bottom: 22px;
        overflow: hidden;
    }}

    .hero::before {{
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 4px;
        background: linear-gradient({CYAN}, {PURPLE});
    }}

    .hero-title {{
        font-size: 30px;
        font-weight: 800;
        color: #0F172A;
        margin: 2px 0 6px 0;
    }}

    .hero-subtitle {{
        font-size: 14px;
        color: {MUTED};
        line-height: 1.65;
        max-width: 900px;
    }}

    .eyebrow {{
        color: {CYAN};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        text-transform: uppercase;
    }}

    .chip {{
        display: inline-block;
        margin-top: 12px;
        padding: 4px 12px;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: {CYAN};
        background: rgba(8, 145, 178, 0.10);
        border: 1px solid rgba(8, 145, 178, 0.35);
        border-radius: 999px;
    }}

    /* ---------- Cards ---------- */
    .panel {{
        background: linear-gradient(180deg, #FFFFFF, #F8FAFD);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.06);
        transition: transform .15s ease, border-color .15s ease,
                    box-shadow .15s ease;
    }}

    .panel:hover {{
        transform: translateY(-2px);
        border-color: #B6C4D9;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10);
    }}

    .panel-label {{
        font-size: 11px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
    }}

    .panel-value {{
        font-size: 28px;
        font-weight: 800;
        color: #0F172A;
        margin-top: 7px;
    }}

    .panel-note {{
        color: {MUTED};
        font-size: 12px;
        margin-top: 5px;
    }}

    .status-dot {{
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: {GREEN};
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.6);
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(5, 150, 105, 0.5); }}
        70%  {{ box-shadow: 0 0 0 8px rgba(5, 150, 105, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(5, 150, 105, 0); }}
    }}

    .small-muted {{ color: {MUTED}; font-size: 12px; }}

    /* Bordered st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px;
        border-color: {BORDER};
    }}

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, #FFFFFF, #F8FAFD);
        border: 1px solid {BORDER};
        border-left: 3px solid {CYAN};
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.06);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 11px;
    }}

    div[data-testid="stMetricValue"] {{
        color: #0F172A;
        font-weight: 800;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid {BORDER};
        background: #FFFFFF;
        color: {TEXT};
        transition: all .15s ease;
    }}

    .stButton > button p {{ color: inherit; }}

    .stButton > button:hover {{
        border-color: {CYAN};
        color: {CYAN};
        transform: translateY(-1px);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #0891B2, #2563EB);
        border: none;
        color: #FFFFFF;
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.25);
    }}

    .stButton > button[kind="primary"] p {{ color: #FFFFFF !important; }}

    .stButton > button[kind="primary"]:hover {{
        color: #FFFFFF;
        filter: brightness(1.08);
        box-shadow: 0 8px 22px rgba(37, 99, 235, 0.35);
    }}

    /* ---------- Inputs ---------- */
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div {{
        background: #FFFFFF !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        color: {TEXT} !important;
    }}

    .stTextArea textarea:focus,
    .stTextInput input:focus {{
        border-color: {CYAN} !important;
        box-shadow: 0 0 0 1px {CYAN} !important;
    }}

    /* ---------- Tables ---------- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
        background: #FFFFFF;
    }}

    /* ---------- Alerts ---------- */
    div[data-testid="stAlert"] {{
        border-radius: 12px;
        border: 1px solid {BORDER};
    }}

    /* ---------- Expander / tabs ---------- */
    details[data-testid="stExpander"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        background: #FFFFFF;
    }}

    button[data-baseweb="tab"] {{ font-weight: 700; }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #FFFFFF, #EEF3FA);
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] {{
        gap: 6px;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] > label {{
        padding: 10px 14px;
        border-radius: 10px;
        border: 1px solid transparent;
        cursor: pointer;
        transition: all .15s ease;
        width: 100%;
    }}

    /* hide the round radio indicator */
    [data-testid="stSidebar"] div[role="radiogroup"] > label
        > div:first-child {{
        display: none;
    }}

    [data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: rgba(8, 145, 178, 0.07);
        border-color: {BORDER};
    }}

    [data-testid="stSidebar"] div[role="radiogroup"]
        > label:has(input:checked) {{
        background: linear-gradient(90deg,
            rgba(8, 145, 178, 0.16), rgba(37, 99, 235, 0.06));
        border-color: rgba(8, 145, 178, 0.45);
        box-shadow: inset 3px 0 0 {CYAN};
    }}

    [data-testid="stSidebar"] div[role="radiogroup"]
        > label:has(input:checked) p {{
        color: #0F172A;
        font-weight: 700;
    }}

    .nav-heading {{
        color: {MUTED};
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 2px;
        margin: 4px 0 10px 4px;
    }}

    hr {{ border-color: {BORDER}; }}

    /* ---------- Scrollbars ---------- */
    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-thumb {{
        background: #C5D1E3;
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: #A9B9D1; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. COMMON UTILITIES
# ============================================================

def panel(label, value, note="", accent=CYAN):
    """Display a dashboard metric card."""
    st.markdown(
        f"""
        <div class="panel" style="border-top: 2px solid {accent};">
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
        font=dict(
            color=TEXT,
            family='-apple-system, "Segoe UI", Roboto, Arial, sans-serif',
        ),
        colorway=[CYAN, PURPLE, BLUE, GREEN, ORANGE, RED],
        margin=dict(l=20, r=20, t=55, b=25),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        title_font=dict(size=15, color=TEXT),
        hoverlabel=dict(
            bgcolor=PANEL,
            bordercolor=BORDER,
            font=dict(color=TEXT),
        ),
        bargap=0.25,
    )

    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.35)",
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )
    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.35)",
        zerolinecolor=BORDER,
        linecolor=BORDER,
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
        mu, _ = model.encode(tensor_data)
        reconstruction = model.decode(mu)

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
        mu, logvar = model.encode(tensor)
        reconstruction = model.decode(mu)

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
# 8. LOGO, NAVIGATION STATE AND COMPACT LAYOUT
# ============================================================

from PIL import Image, ImageDraw

LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "m2_ai_logo.png",
)


@st.cache_data(show_spinner=False)
def load_logo(path, modified_time):
    """Return the logo with its white background made transparent."""
    img = Image.open(path).convert("RGBA")
    img.thumbnail((480, 480))

    # Keep the file as it is if it already has real transparency.
    if img.getchannel("A").getextrema()[0] < 250:
        return img

    width, height = img.size
    corners = [
        (0, 0),
        (width - 1, 0),
        (0, height - 1),
        (width - 1, height - 1),
    ]

    # Flood-fill only the background that touches the image corners, so
    # white areas inside the logo itself are preserved.
    for corner in corners:
        r, g, b, _ = img.getpixel(corner)
        if r > 225 and g > 225 and b > 225:
            ImageDraw.floodfill(
                img,
                corner,
                (255, 255, 255, 0),
                thresh=60,
            )

    return img


def get_logo():
    if not os.path.exists(LOGO_PATH):
        return None
    try:
        return load_logo(LOGO_PATH, os.path.getmtime(LOGO_PATH))
    except Exception:
        return None


NAV_ICONS = {
    "VAE Laboratory": "🧬",
    "Naive Bayes Laboratory": "📊",
    "HMM Laboratory": "🔗",
    "Agentic RAG": "🔎",
    "RAG Evaluation Lab": "🧪",
}

PAGES = list(NAV_ICONS.keys())

PAGE_BLURBS = {
    "VAE Laboratory": (
        "Reconstruct synthetic telemetry and flag unusual readings "
        "with a variational autoencoder."
    ),
    "Naive Bayes Laboratory": (
        "Classify maintenance queries and see how the predicted "
        "category influences retrieval."
    ),
    "HMM Laboratory": (
        "Decode hidden equipment states from observed symptoms "
        "using the Viterbi algorithm."
    ),
    "Agentic RAG": (
        "Retrieve technical evidence with TF-IDF and "
        "category-aware reranking."
    ),
    "RAG Evaluation Lab": (
        "Compare retrieval strategies against labelled questions "
        "using Recall@3 and MRR."
    ),
}

# Contextual links to the relevant sections of the HTML manual
MANUAL_URL = "/app/static/user_manual.html"

MANUAL_SECTIONS = {
    "VAE Laboratory": "vae-lab",
    "Naive Bayes Laboratory": "nb-lab",
    "HMM Laboratory": "hmm-lab",
    "Agentic RAG": "agentic-rag",
    "RAG Evaluation Lab": "rag-eval",
}



def nav_label(page_name):
    return f"{NAV_ICONS[page_name]}  {page_name}"


# The sidebar menu and the top tabs share one page state.
if "page" not in st.session_state:
    st.session_state.page = PAGES[0]
if "nav_side" not in st.session_state:
    st.session_state.nav_side = st.session_state.page
if "nav_top" not in st.session_state:
    st.session_state.nav_top = st.session_state.page


def sync_from_sidebar():
    st.session_state.page = st.session_state.nav_side
    st.session_state.nav_top = st.session_state.nav_side


def sync_from_top():
    choice = st.session_state.nav_top
    if choice is None:
        # Clicking the active tab again must not leave the app pageless.
        st.session_state.nav_top = st.session_state.page
    else:
        st.session_state.page = choice
        st.session_state.nav_side = choice


# Compact layout and top-tab styling (overrides the theme CSS above).
st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 2.4rem;
        padding-bottom: 1rem;
    }}

    div[data-testid="stVerticalBlock"] {{ gap: 0.7rem; }}

    .hero {{
        padding: 18px 24px;
        margin-bottom: 0;
        border-radius: 14px;
    }}

    .hero-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 24px;
    }}

    .hero-title {{
        font-size: 28px;
        margin: 0;
    }}

    .hero-subtitle {{
        font-size: 13px;
        line-height: 1.5;
        max-width: 55%;
        text-align: right;
    }}

    h3 {{
        font-size: 1rem !important;
        margin-top: 0.3rem !important;
        margin-bottom: 0.1rem !important;
    }}

    h4 {{
        margin: 0 0 0.2rem 0 !important;
        font-size: 0.95rem !important;
    }}

    .panel {{
        padding: 12px 16px;
        margin-bottom: 0;
    }}

    .panel-value {{
        font-size: 22px;
        margin-top: 4px;
    }}

    div[data-testid="stMetric"] {{ padding: 8px 12px; }}
    div[data-testid="stMetricValue"] {{ font-size: 1.35rem; }}

    /* ---------- Top navigation tabs ---------- */
    div[data-testid="stButtonGroup"] {{
        width: 100%;
        gap: 6px;
        flex-wrap: nowrap;
        border-bottom: 1px solid {BORDER};
    }}

    button[data-testid="stBaseButton-segmented_control"],
    button[data-testid="stBaseButton-segmented_controlActive"] {{
        flex: 1 1 0;
        justify-content: center;
        padding: 8px 14px;
        border-radius: 10px 10px 0 0 !important;
        border: 1px solid {BORDER} !important;
        border-bottom: 3px solid transparent !important;
        background: #FFFFFF !important;
        font-weight: 700;
    }}

    button[data-testid="stBaseButton-segmented_control"] p {{
        color: {MUTED} !important;
    }}

    button[data-testid="stBaseButton-segmented_control"]:hover {{
        background: rgba(8, 145, 178, 0.07) !important;
    }}

    button[data-testid="stBaseButton-segmented_controlActive"] {{
        background: linear-gradient(180deg,
            rgba(8, 145, 178, 0.14), #FFFFFF) !important;
        border-bottom: 3px solid {CYAN} !important;
    }}

    button[data-testid="stBaseButton-segmented_controlActive"] p {{
        color: {CYAN} !important;
    }}

    /* Metric text must wrap instead of being cut off with "..." */
    div[data-testid="stMetricValue"] > div,
    div[data-testid="stMetricLabel"] p,
    div[data-testid="stMetricLabel"] > div {{
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere;
    }}

    .st-key-vae_summary div[data-testid="stMetricValue"] {{
        font-size: 1.15rem;
    }}

    .st-key-vae_summary div[data-testid="stMetricLabel"] {{
        font-size: 0.7rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 9. SIDEBAR AND COMMON HEADER
# ============================================================

with st.sidebar:
    sidebar_logo = get_logo()

    if sidebar_logo is not None:
        st.image(sidebar_logo, width=150)

        st.markdown(
            f"""
            <div style="font-size: 11px; color: {MUTED};
                        letter-spacing: 2px; margin: -2px 0 16px 4px;">
                MODEL LABORATORY
            </div>
            <div class="nav-heading">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 8px 4px 18px 4px;">
                <div style="font-size: 30px; font-weight: 900;
                            background: linear-gradient(90deg,
                                {CYAN}, {PURPLE});
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;">
                    M² AI
                </div>
                <div style="font-size: 11px; color: {MUTED};
                            letter-spacing: 2px; margin-top: 2px;">
                    MODEL LABORATORY
                </div>
            </div>
            <div class="nav-heading">NAVIGATION</div>
            """,
            unsafe_allow_html=True,
        )

    st.radio(
        "Select laboratory",
        PAGES,
        format_func=nav_label,
        key="nav_side",
        on_change=sync_from_sidebar,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown(
        f"""
        <div class="nav-heading">SYSTEM STATUS</div>
        <div style="color: {GREEN}; font-size: 13px;
                    margin-left: 4px;">
            <span class="status-dot"></span>Application loaded
        </div>
        <div style="color: {MUTED}; font-size: 12px;
                    margin: 8px 0 0 4px; line-height: 1.7;">
            Synthetic demonstration data<br>
            Local model inference
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.caption("Technical AI Model Laboratory")
    st.caption("Educational prototype • Synthetic data only")

page = st.session_state.page

logo_col, hero_col = st.columns([2, 12], vertical_alignment="center")

with logo_col:
    logo_image = get_logo()
    if logo_image is not None:
        st.image(logo_image, width=150)
    else:
        st.markdown(
            f"""
            <div style="font-size: 56px; font-weight: 900;
                        color: {CYAN}; text-align: center;">
                M²
            </div>
            """,
            unsafe_allow_html=True,
        )

with hero_col:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-row">
                <div>
                    <div class="eyebrow">
                        M² AI · TECHNICAL AI MODEL LABORATORY
                    </div>
                    <div class="hero-title">
                        {NAV_ICONS[page]} {page}
                    </div>
                </div>
                
            <div class="hero-subtitle">{PAGE_BLURBS[page]}</div>
            <a class="manual-link"
               href="{MANUAL_URL}#{MANUAL_SECTIONS[page]}"
               target="_blank"
               rel="noopener noreferrer"
               style="display:inline-block; margin-top:10px; font-size:13px;
                      font-weight:700; color:#0891B2; white-space:nowrap;">
                📖 User Manual
            </a>

               
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.segmented_control(
    "Navigation",
    PAGES,
    format_func=nav_label,
    selection_mode="single",
    key="nav_top",
    on_change=sync_from_top,
    label_visibility="collapsed",
)


# ============================================================
# 10. VAE LABORATORY
# ============================================================

if page == "VAE Laboratory":

    model, scaler, normal_data, threshold, train_errors = train_vae()

    control_col, gauge_col, summary_col = st.columns(
        [1.1, 1.0, 1.5],
        gap="medium",
    )

    # --------------------------------------------------------
    # Live telemetry controls
    # --------------------------------------------------------
    with control_col:
        with st.container(border=True):
            st.markdown("#### Live telemetry input")

            slider_left, slider_right = st.columns(2)

            with slider_left:
                temperature = st.slider(
                    "Temperature (°C)",
                    min_value=20.0,
                    max_value=120.0,
                    value=55.0,
                    step=1.0,
                    key="temp",
                )

                vibration = st.slider(
                    "Vibration",
                    min_value=0.0,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key="vib",
                )

            with slider_right:
                voltage = st.slider(
                    "Voltage (V)",
                    min_value=15.0,
                    max_value=40.0,
                    value=28.0,
                    step=0.5,
                    key="volt",
                )

                packet_loss = st.slider(
                    "Packet loss (%)",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.6,
                    step=0.1,
                    key="packet",
                )

            def reset_telemetry():
                st.session_state.temp = 55.0
                st.session_state.vib = 1.0
                st.session_state.volt = 28.0
                st.session_state.packet = 0.6

            st.button(
                "↻ Reset to Normal Values",
                key="reset_telemetry_button",
                on_click=reset_telemetry,
                width="stretch",
            )

    telemetry = [
        temperature,
        vibration,
        voltage,
        packet_loss,
    ]

    analysis = analyze_telemetry(telemetry)

    error = analysis["error"]
    threshold_value = analysis["threshold"]
    is_anomaly = analysis["anomaly"]

    status_color = RED if is_anomaly else GREEN
    status_text = "ANOMALY DETECTED" if is_anomaly else "NORMAL RANGE"
    gauge_max = max(threshold_value * 2, error * 1.2, 0.01)

    # --------------------------------------------------------
    # Reconstruction error gauge
    # --------------------------------------------------------
    with gauge_col:
        with st.container(border=True):
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=error,
                    number={
                        "valueformat": ".4f",
                        "font": {"size": 32, "color": TEXT},
                    },
                    title={
                        "text": "Reconstruction error",
                        "font": {"size": 13, "color": MUTED},
                    },
                    gauge={
                        "axis": {
                            "range": [0, gauge_max],
                            "tickcolor": MUTED,
                        },
                        "bar": {"color": status_color},
                        "bgcolor": "#FFFFFF",
                        "bordercolor": BORDER,
                        "steps": [
                            {
                                "range": [0, threshold_value],
                                "color": "#D1FAE5",
                            },
                            {
                                "range": [threshold_value, gauge_max],
                                "color": "#FFE4E6",
                            },
                        ],
                        "threshold": {
                            "line": {"color": ORANGE, "width": 4},
                            "thickness": 0.8,
                            "value": threshold_value,
                        },
                    },
                )
            )

            fig_gauge.update_layout(
                height=215,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": TEXT},
                margin={"l": 25, "r": 25, "t": 45, "b": 5},
            )

            st.plotly_chart(
                fig_gauge,
                width="stretch",
                config={"displayModeBar": False},
            )

            st.markdown(
                f"""
                <div style="
                    text-align:center;
                    padding:6px;
                    border:1px solid {status_color};
                    border-radius:10px;
                    color:{status_color};
                    font-weight:800;
                    letter-spacing:1px;
                ">
                    {status_text}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(
                f"Training threshold: {threshold_value:.4f} "
                "(95th percentile of synthetic training errors)"
            )

    # --------------------------------------------------------
    # Summary metrics and telemetry table
    # --------------------------------------------------------
    reconstruction_df = pd.DataFrame({
        "Feature": FEATURES,
        "Input": telemetry,
        "Reconstruction": analysis["reconstruction"],
    })

    with summary_col:
        with st.container(border=True, key="vae_summary"):
            st.markdown("#### Model summary")

            m1, m2, m3 = st.columns(3)

            m1.metric("Error", f"{error:.4f}")
            m2.metric("Threshold", f"{threshold_value:.4f}")
            m3.metric("Result", "Anomaly" if is_anomaly else "Normal")

            if is_anomaly:
                st.warning(
                    "The input exceeds the illustrative "
                    "reconstruction-error threshold."
                )
            else:
                st.success(
                    "The input is within the illustrative "
                    "reconstruction-error threshold."
                )

            st.dataframe(
                reconstruction_df.round(4),
                width="stretch",
                hide_index=True,
                height=178,
            )

    # --------------------------------------------------------
    # Diagnostics row
    # --------------------------------------------------------
    reconstruction_long = reconstruction_df.melt(
        id_vars="Feature",
        var_name="Measurement",
        value_name="Value",
    )

    chart_a, chart_b, chart_c = st.columns(
        [1.3, 0.8, 1.1],
        gap="medium",
    )

    with chart_a:
        with st.container(border=True):
            fig_reconstruction = px.bar(
                reconstruction_long,
                x="Feature",
                y="Value",
                color="Measurement",
                barmode="group",
                title="Input vs reconstruction",
                color_discrete_map={
                    "Input": CYAN,
                    "Reconstruction": PURPLE,
                },
            )

            st.plotly_chart(
                style_plot(fig_reconstruction, 270),
                width="stretch",
            )

    with chart_b:
        with st.container(border=True):
            latent_df = pd.DataFrame({
                "Latent dimension": ["z1", "z2"],
                "Value": analysis["latent"],
            })

            fig_latent = px.bar(
                latent_df,
                x="Latent dimension",
                y="Value",
                color="Latent dimension",
                title="Encoder latent mean",
                color_discrete_map={
                    "z1": CYAN,
                    "z2": PURPLE,
                },
            )

            fig_latent.update_layout(showlegend=False)

            st.plotly_chart(
                style_plot(fig_latent, 270),
                width="stretch",
            )

    with chart_c:
        with st.container(border=True):
            fig_errors = px.histogram(
                x=train_errors,
                nbins=35,
                title="Training reconstruction errors",
                labels={"x": "Reconstruction error"},
                color_discrete_sequence=[BLUE],
            )

            fig_errors.add_vline(
                x=threshold_value,
                line_dash="dash",
                line_color=ORANGE,
                annotation_text="Threshold",
            )

            fig_errors.add_vline(
                x=error,
                line_dash="solid",
                line_color=RED,
                annotation_text="Current input",
            )

            st.plotly_chart(
                style_plot(fig_errors, 270),
                width="stretch",
            )

    st.caption(
        "Educational demonstration using synthetic telemetry. "
        "A reconstruction error above the learned threshold indicates "
        "an unusual input under this model; it is not a validated "
        "equipment-safety diagnosis."
    )


# ============================================================
# 11. NAIVE BAYES LABORATORY
# ============================================================

elif page == "Naive Bayes Laboratory":

    model = build_nb_model()

    left, right = st.columns([1, 2.3], gap="medium")

    with left:
        with st.container(border=True):
            query = st.text_area(
                "Enter a technical symptom or maintenance query",
                value="The equipment is restarting unexpectedly due to unstable power.",
                height=120,
            )

            run_classify = st.button(
                "Classify query",
                type="primary",
                width="stretch",
            )

        with st.expander("How Naive Bayes works", expanded=True):
            st.write(
                "Naive Bayes estimates the probability of a class given "
                "the observed words. It applies Bayes' theorem and "
                "assumes that features are conditionally independent "
                "given the class."
            )

            st.latex(
                r"P(C\mid x_1,\ldots,x_n)"
                r"\propto P(C)\prod_{i=1}^{n}P(x_i\mid C)"
            )

            st.caption(
                "This model uses a small, manually authored synthetic "
                "training set. Its probabilities are not calibrated "
                "operational confidence."
            )

    with right:
        if run_classify:
            predicted, probabilities = predict_category(query)

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
                    accent=PURPLE,
                )

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

            fig.update_layout(coloraxis_showscale=False)

            with st.container(border=True):
                st.plotly_chart(
                    style_plot(fig, 210),
                    width="stretch",
                )

            st.markdown("#### Retrieved documents")

            results = category_rank(
                query,
                predicted,
                k=3,
            )

            doc_cols = st.columns(len(results))

            for col, (rank, doc) in zip(
                doc_cols,
                enumerate(results, start=1),
            ):
                with col:
                    with st.container(border=True):
                        st.markdown(
                            f"**#{rank} — {doc['title']}**"
                        )
                        st.caption(
                            f"{doc['id']} • {doc['category']}"
                        )
                        st.write(doc["text"])
                        st.caption(
                            f"Similarity `{doc['score']:.3f}` • "
                            f"Adjusted `{doc['adjusted_score']:.3f}`"
                        )

        else:
            with st.container(border=True):
                st.markdown("#### Model training data")

                training_df = pd.DataFrame(
                    NB_TRAINING_DATA,
                    columns=["Example text", "Category"],
                )

                st.dataframe(
                    training_df,
                    width="stretch",
                    hide_index=True,
                    height=320,
                )

            st.info(
                "Enter a query and click Classify query to see how the "
                "model predicts a category and how that category can "
                "influence retrieval."
            )


# ============================================================
# 12. HMM LABORATORY
# ============================================================

elif page == "HMM Laboratory":

    default_sequence = [
        "Normal temperature",
        "Elevated temperature",
        "High packet loss",
        "Repeated restart",
    ]

    with st.container(border=True):
        selected_sequence = st.multiselect(
            "Observation sequence, in time order",
            options=HMM_OBSERVATIONS,
            default=default_sequence,
            help=(
                "Select a sequence of observable symptoms. The HMM "
                "estimates the most likely sequence of hidden "
                "equipment states."
            ),
        )

    if not selected_sequence:
        st.warning("Select at least one observation to run Viterbi.")
    else:
        states, final_probs = viterbi(selected_sequence)

        state_colors = {
            "Normal": GREEN,
            "Degraded": ORANGE,
            "Fault": RED,
        }

        timeline_col, prob_col, table_col = st.columns(
            [1.2, 1, 1],
            gap="medium",
        )

        with timeline_col:
            with st.container(border=True):
                fig = go.Figure()

                for state_name in HMM_STATES:
                    times = [
                        i + 1
                        for i, state in enumerate(states)
                        if state == state_name
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
                    showlegend=False,
                )

                st.plotly_chart(
                    style_plot(fig, 290),
                    width="stretch",
                )

        with prob_col:
            with st.container(border=True):
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
                    color_discrete_map=state_colors,
                    title="Final-step state scores (normalized)",
                )

                fig.update_layout(showlegend=False)

                st.plotly_chart(
                    style_plot(fig, 290),
                    width="stretch",
                )

        with table_col:
            with st.container(border=True):
                st.markdown("#### Inferred state sequence")

                state_df = pd.DataFrame({
                    "Time step": np.arange(1, len(states) + 1),
                    "Observation": selected_sequence,
                    "Inferred state": states,
                })

                st.dataframe(
                    state_df,
                    width="stretch",
                    hide_index=True,
                    height=215,
                )

        trans_col, emis_col = st.columns(2, gap="medium")

        with trans_col:
            with st.container(border=True):
                st.markdown("#### HMM transition matrix")

                transition_df = pd.DataFrame(
                    HMM_TRANSITION,
                    index=HMM_STATES,
                    columns=HMM_STATES,
                )

                st.dataframe(
                    transition_df.style.format("{:.2f}"),
                    width="stretch",
                    height=148,
                )

        with emis_col:
            with st.container(border=True):
                st.markdown("#### HMM emission matrix")

                emission_df = pd.DataFrame(
                    HMM_EMISSION,
                    index=HMM_STATES,
                    columns=HMM_OBSERVATIONS,
                )

                st.dataframe(
                    emission_df.style.format("{:.2f}"),
                    width="stretch",
                    height=148,
                )

    st.caption(
        "Teaching point: the Viterbi algorithm finds the most likely "
        "hidden-state sequence given the observations and the model "
        "parameters. These transition and emission probabilities are "
        "illustrative, not learned from real equipment histories."
    )


# ============================================================
# 13. AGENTIC RAG
# ============================================================

elif page == "Agentic RAG":

    left, right = st.columns([1, 2.4], gap="medium")

    with left:
        with st.container(border=True):
            query = st.text_area(
                "Ask a technical question",
                value="The equipment is restarting unexpectedly. What should I investigate?",
                height=110,
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
            )

            run_retrieval = st.button(
                "Run retrieval",
                type="primary",
                width="stretch",
            )

    with right:
        tab_evidence, tab_scores, tab_classifier, tab_arch = st.tabs([
            "Retrieved evidence",
            "Score comparison",
            "Classifier probabilities",
            "Prototype architecture",
        ])

        if run_retrieval:
            predicted_category, probabilities = predict_category(query)

            if method == "Baseline TF-IDF":
                results = retrieve(query, k=k)
            else:
                results = category_rank(
                    query,
                    predicted_category,
                    k=k,
                )

        with tab_evidence:
            if run_retrieval:
                st.write(
                    f"**Predicted category:** "
                    f"`{predicted_category}`"
                )

                for start in range(0, len(results), 3):
                    row_docs = results[start:start + 3]
                    row_cols = st.columns(3)

                    for col, (offset, doc) in zip(
                        row_cols,
                        enumerate(row_docs),
                    ):
                        rank = start + offset + 1
                        score = doc.get(
                            "adjusted_score",
                            doc["score"],
                        )

                        with col:
                            with st.container(border=True):
                                st.markdown(
                                    f"**{rank}. {doc['title']}**"
                                )
                                st.caption(
                                    f"{doc['id']} • "
                                    f"{doc['category']} • "
                                    f"Score {score:.3f}"
                                )
                                st.write(doc["text"])
            else:
                st.info(
                    "Ask a question and click Run retrieval to see the "
                    "retrieved evidence."
                )

        with tab_scores:
            if run_retrieval:
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
                    style_plot(fig, 300),
                    width="stretch",
                )
            else:
                st.info("Run retrieval to see the score comparison.")

        with tab_classifier:
            if run_retrieval:
                probability_df = pd.DataFrame({
                    "Category": list(probabilities.keys()),
                    "Probability": list(probabilities.values()),
                }).sort_values("Probability", ascending=False)

                st.dataframe(
                    probability_df,
                    width="stretch",
                    hide_index=True,
                    height=150,
                )
            else:
                st.info("Run retrieval to see classifier probabilities.")

        with tab_arch:
            st.graphviz_chart(
                """
                digraph RAG {
                    rankdir=LR;
                    bgcolor="transparent";
                    node [
                        shape=box,
                        style="rounded,filled",
                        fillcolor="#FFFFFF",
                        color="#0891B2",
                        fontcolor="#0F172A",
                        fontname="Arial"
                    ];
                    edge [
                        color="#64748B",
                        fontcolor="#0F172A",
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
                "This is a retrieval prototype, not a complete "
                "autonomous agent or a production generative-answer "
                "system. It retrieves evidence from synthetic documents "
                "but does not call an LLM to generate a grounded answer."
            )


# ============================================================
# 14. RAG EVALUATION LAB
# ============================================================

elif page == "RAG Evaluation Lab":

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

    # --------------------------------------------------------
    # Top row: control + overall metrics
    # --------------------------------------------------------
    ctrl_col, m_col1, m_col2, m_col3, m_col4 = st.columns(
        [1.5, 1, 1, 1, 1],
        gap="small",
        vertical_alignment="center",
    )

    with ctrl_col:
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

    metrics_df, details_df = evaluate_with_bonus(bonus)

    baseline_recall = metrics_df["Baseline Recall@3"].mean()
    category_recall = metrics_df["Category Recall@3"].mean()
    baseline_mrr = metrics_df["Baseline RR"].mean()
    category_mrr = metrics_df["Category RR"].mean()

    m_col1.metric(
        "Baseline Recall@3",
        f"{baseline_recall:.1%}",
    )

    m_col2.metric(
        "Category Recall@3",
        f"{category_recall:.1%}",
        delta=f"{category_recall - baseline_recall:+.1%}",
    )

    m_col3.metric(
        "Baseline MRR",
        f"{baseline_mrr:.3f}",
    )

    m_col4.metric(
        "Category MRR",
        f"{category_mrr:.3f}",
        delta=f"{category_mrr - baseline_mrr:+.3f}",
    )

    changed_count = int(details_df["Ranking changed"].sum())

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

    with st.expander("About this experiment"):
        st.markdown(
            """
            This experiment compares two retrieval strategies over a
            fixed set of labelled synthetic technical questions:

            - **Baseline RAG:** TF-IDF similarity only.
            - **Category-aware RAG:** Naive Bayes predicts a category,
              and matching documents receive a ranking bonus.
            """
        )

        st.warning(
            "The relevance labels are manually assigned teaching "
            "labels. Review them if a question has more than one valid "
            "answer."
        )

    tab_overview, tab_ranking, tab_inspect, tab_truth = st.tabs([
        "Overview",
        "Ranking changes & failure analysis",
        "Inspect a question",
        "Ground-truth labels",
    ])

    # --------------------------------------------------------
    # Overview tab
    # --------------------------------------------------------
    with tab_overview:
        chart_left, chart_right = st.columns(2, gap="medium")

        with chart_left:
            with st.container(border=True):
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
                    style_plot(fig, 300),
                    width="stretch",
                )

        with chart_right:
            with st.container(border=True):
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
                    title="Recall@3 for each question",
                    color_discrete_sequence=[BLUE, CYAN],
                )

                fig.update_yaxes(showticklabels=False, title_text="")

                st.plotly_chart(
                    style_plot(fig, 300),
                    width="stretch",
                )

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Rankings changed",
            f"{changed_count}/{len(EVALUATION_SET)}",
        )

        s2.metric("New successful hits", improved_count)

        s3.metric("Lost successful hits", regressed_count)

    # --------------------------------------------------------
    # Ranking changes tab
    # --------------------------------------------------------
    with tab_ranking:
        st.dataframe(
            details_df,
            width="stretch",
            hide_index=True,
            height=250,
        )

        with st.expander("View question-level metric details"):
            st.dataframe(
                metrics_df,
                width="stretch",
                hide_index=True,
                height=250,
            )

        st.caption(
            "Interpretation: Recall@3 measures the fraction of labelled "
            "relevant documents found in the top three. MRR rewards "
            "finding the first relevant result near the top. A ranking "
            "change is not automatically an improvement; use the "
            "relevance labels to judge it."
        )

    # --------------------------------------------------------
    # Inspect tab
    # --------------------------------------------------------
    with tab_inspect:
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

        left, right = st.columns(2, gap="medium")

        with left:
            with st.container(border=True):
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
            with st.container(border=True):
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

    # --------------------------------------------------------
    # Ground truth tab
    # --------------------------------------------------------
    with tab_truth:
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
            width="stretch",
            hide_index=True,
            height=250,
        )

        st.caption(
            "Scope: six synthetic documents and six manually labelled "
            "questions. This is an educational experiment, not a "
            "statistically robust benchmark. Results depend on the "
            "labels, training examples, similarity scores, and "
            "category bonus."
        )


# ============================================================
# 15. FOOTER
# ============================================================

st.markdown(
    f"""
    <div style="text-align: center; padding: 4px 0 0 0;
                font-size: 11px; color: {MUTED};">
        TECHNICAL AI MODEL LABORATORY &nbsp;·&nbsp;
        VAE • Naive Bayes • HMM • RAG • Evaluation
    </div>
    """,
    unsafe_allow_html=True,
)
