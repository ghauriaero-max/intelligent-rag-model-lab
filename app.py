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
    /* ---------- Base ---------- */
    html, body, .stApp {{
        font-family: -apple-system, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }}

    .stApp {{
        background:
            radial-gradient(900px 500px at 8% -10%,
                rgba(34, 211, 238, 0.10), transparent 60%),
            radial-gradient(800px 500px at 100% 0%,
                rgba(167, 139, 250, 0.10), transparent 55%),
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
        color: #CBD5E1 !important;
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
            linear-gradient(120deg, rgba(34, 211, 238, 0.07),
                            rgba(167, 139, 250, 0.07)),
            #0F172A;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
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
        color: #F8FAFC;
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
        background: rgba(34, 211, 238, 0.10);
        border: 1px solid rgba(34, 211, 238, 0.35);
        border-radius: 999px;
    }}

    /* ---------- Cards ---------- */
    .panel {{
        background: linear-gradient(180deg, #131C2E, {PANEL});
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 15px;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
        transition: transform .15s ease, border-color .15s ease;
    }}

    .panel:hover {{
        transform: translateY(-2px);
        border-color: #35476A;
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
        width: 8px; height: 8px;
        border-radius: 50%;
        background: {GREEN};
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.6);
        animation: pulse 2s infinite;
    }}

    @keyframes pulse {{
        0%   {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.55); }}
        70%  {{ box-shadow: 0 0 0 8px rgba(52, 211, 153, 0); }}
        100% {{ box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }}
    }}

    .small-muted {{ color: {MUTED}; font-size: 12px; }}

    /* Bordered st.container(border=True) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        border-radius: 14px;
        border-color: {BORDER};
    }}

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, #131C2E, {PANEL});
        border: 1px solid {BORDER};
        border-left: 3px solid {CYAN};
        padding: 14px 16px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.22);
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 0.8px;
        font-size: 11px;
    }}

    div[data-testid="stMetricValue"] {{
        color: #F8FAFC;
        font-weight: 800;
    }}

    /* ---------- Buttons ---------- */
    .stButton > button {{
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid {BORDER};
        background: #131C2E;
        color: {TEXT};
        transition: all .15s ease;
    }}

    .stButton > button:hover {{
        border-color: {CYAN};
        color: {CYAN};
        transform: translateY(-1px);
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        border: none;
        color: #04111C;
        box-shadow: 0 6px 18px rgba(34, 211, 238, 0.25);
    }}

    .stButton > button[kind="primary"]:hover {{
        color: #04111C;
        filter: brightness(1.1);
        box-shadow: 0 8px 22px rgba(34, 211, 238, 0.35);
    }}

    /* ---------- Inputs ---------- */
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="select"] > div {{
        background: #0E1626 !important;
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
        background: #0E1626;
    }}

    button[data-baseweb="tab"] {{ font-weight: 700; }}

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0C1424, #080D18);
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
        background: rgba(34, 211, 238, 0.07);
        border-color: {BORDER};
    }}

    [data-testid="stSidebar"] div[role="radiogroup"]
        > label:has(input:checked) {{
        background: linear-gradient(90deg,
            rgba(34, 211, 238, 0.18), rgba(96, 165, 250, 0.08));
        border-color: rgba(34, 211, 238, 0.5);
        box-shadow: inset 3px 0 0 {CYAN};
    }}

    [data-testid="stSidebar"] div[role="radiogroup"]
        > label:has(input:checked) p {{
        color: #F8FAFC;
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
        background: #263449;
        border-radius: 8px;
    }}
    ::-webkit-scrollbar-thumb:hover {{ background: #35476A; }}
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
        gridcolor="rgba(38, 52, 73, 0.6)",
        zerolinecolor=BORDER,
        linecolor=BORDER,
    )
    fig.update_yaxes(
        gridcolor="rgba(38, 52, 73, 0.6)",
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

NAV_ICONS = {
    "VAE Laboratory": "🧬",
    "Naive Bayes Laboratory": "📊",
    "HMM Laboratory": "🔗",
    "Agentic RAG": "🔎",
    "RAG Evaluation Lab": "🧪",
}

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

LOGO_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "m2_ai_logo.png",
)

with st.sidebar:
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

    page = st.radio(
        "Select laboratory",
        list(NAV_ICONS.keys()),
        format_func=lambda p: f"{NAV_ICONS[p]}   {p}",
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


# ============================================================
# 9. COMMON PAGE HEADER (LOGO + HERO BANNER)
# ============================================================

logo_col, hero_col = st.columns([1, 8], vertical_alignment="center")

with logo_col:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=96)
    else:
        st.markdown(
            f"""
            <div style="font-size: 34px; font-weight: 900;
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
            <div class="eyebrow">
                M² AI · INTELLIGENT SYSTEMS · EXPERIMENTAL LAB
            </div>
            <div class="hero-title">Technical AI Model Laboratory</div>
            <div class="hero-subtitle">
                {PAGE_BLURBS[page]}
            </div>
            <span class="chip">
                {NAV_ICONS[page]} &nbsp;{page}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# 10. VAE LABORATORY — REDESIGNED DASHBOARD
# ============================================================

if page == "VAE Laboratory":

    st.markdown("## VAE Laboratory")
    st.caption(
        "VARIATIONAL AUTOENCODER  /  TELEMETRY RECONSTRUCTION  /  "
        "ANOMALY DETECTION"
    )

    # Keep the existing model and analysis functions unchanged.
    model, scaler, normal_data, threshold, train_errors = train_vae()

    # --------------------------------------------------------
    # Live telemetry controls
    # --------------------------------------------------------
    st.markdown("### LIVE TELEMETRY INPUT")

    control_col, gauge_col = st.columns([1.05, 1.5], gap="large")

    with control_col:
        with st.container(border=True):
            st.markdown("#### Telemetry controls")
            st.caption("Adjust synthetic equipment readings.")

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

    # Existing model inference.
    analysis = analyze_telemetry(telemetry)

    error = analysis["error"]
    threshold_value = analysis["threshold"]
    is_anomaly = analysis["anomaly"]

    # --------------------------------------------------------
    # Reconstruction error gauge
    # --------------------------------------------------------
    with gauge_col:
        with st.container(border=True):
            st.markdown("#### Reconstruction error")

            status_color = "#FB7185" if is_anomaly else "#34D399"
            status_text = (
                "ANOMALY DETECTED"
                if is_anomaly
                else "NORMAL RANGE"
            )

            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=error,
                    number={
                        "valueformat": ".4f",
                        "font": {
                            "size": 36,
                            "color": "#E5E7EB",
                        },
                    },
                    title={
                        "text": "Reconstruction error",
                        "font": {
                            "size": 14,
                            "color": "#94A3B8",
                        },
                    },
                    gauge={
                        "axis": {
                            "range": [
                                0,
                                max(
                                    threshold_value * 2,
                                    error * 1.2,
                                    0.01,
                                ),
                            ],
                            "tickcolor": "#94A3B8",
                        },
                        "bar": {"color": status_color},
                        "bgcolor": "#111827",
                        "bordercolor": "#263449",
                        "steps": [
                            {
                                "range": [
                                    0,
                                    threshold_value,
                                ],
                                "color": "#12352F",
                            },
                            {
                                "range": [
                                    threshold_value,
                                    max(
                                        threshold_value * 2,
                                        error * 1.2,
                                        0.01,
                                    ),
                                ],
                                "color": "#3A202B",
                            },
                        ],
                        "threshold": {
                            "line": {
                                "color": "#FBBF24",
                                "width": 4,
                            },
                            "thickness": 0.8,
                            "value": threshold_value,
                        },
                    },
                )
            )

            fig_gauge.update_layout(
                height=275,
                paper_bgcolor="rgba(0,0,0,0)",
                font={"color": "#E5E7EB"},
                margin={"l": 25, "r": 25, "t": 55, "b": 15},
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
                    padding:10px;
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
    # Summary metrics
    # --------------------------------------------------------
    st.markdown("### MODEL SUMMARY")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Reconstruction error",
            f"{error:.4f}",
        )

    with m2:
        st.metric(
            "Training threshold",
            f"{threshold_value:.4f}",
        )

    with m3:
        st.metric(
            "Assessment",
            "Anomaly" if is_anomaly else "Normal",
        )

    if is_anomaly:
        st.warning(
            "The input exceeds the illustrative reconstruction-error "
            "threshold."
        )
    else:
        st.success(
            "The input is within the illustrative reconstruction-error "
            "threshold."
        )

    # --------------------------------------------------------
    # Input versus reconstruction
    # --------------------------------------------------------
    st.markdown("### INPUT VS RECONSTRUCTION")

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

    left, right = st.columns([1.6, 1], gap="large")

    with left:
        with st.container(border=True):
            fig_reconstruction = px.bar(
                reconstruction_long,
                x="Feature",
                y="Value",
                color="Measurement",
                barmode="group",
                title="Telemetry compared with VAE output",
                color_discrete_map={
                    "Input": "#22D3EE",
                    "Reconstruction": "#A78BFA",
                },
            )

            st.plotly_chart(
                style_plot(fig_reconstruction, 360),
                width="stretch",
            )

    with right:
        with st.container(border=True):
            st.markdown("#### Telemetry summary")

            st.dataframe(
                reconstruction_df.round(4),
                width="stretch",
                hide_index=True,
            )

    # --------------------------------------------------------
    # Latent representation and training errors
    # --------------------------------------------------------
    st.markdown("### MODEL DIAGNOSTICS")

    latent_col, error_col = st.columns(2, gap="large")

    with latent_col:
        with st.container(border=True):
            st.markdown("#### Latent representation")

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
                    "z1": "#22D3EE",
                    "z2": "#A78BFA",
                },
            )

            st.plotly_chart(
                style_plot(fig_latent, 310),
                width="stretch",
            )

    with error_col:
        with st.container(border=True):
            st.markdown("#### Training reconstruction errors")

            fig_errors = px.histogram(
                x=train_errors,
                nbins=35,
                title="Synthetic normal training distribution",
                labels={"x": "Reconstruction error"},
                color_discrete_sequence=["#60A5FA"],
            )

            fig_errors.add_vline(
                x=threshold_value,
                line_dash="dash",
                line_color="#FBBF24",
                annotation_text="Threshold",
            )

            fig_errors.add_vline(
                x=error,
                line_dash="solid",
                line_color="#FB7185",
                annotation_text="Current input",
            )

            st.plotly_chart(
                style_plot(fig_errors, 310),
                width="stretch",
            )

    st.info(
        "Educational demonstration using synthetic telemetry. "
        "A reconstruction error above the learned threshold indicates "
        "an unusual input under this model; it is not a validated "
        "equipment-safety diagnosis."
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
