import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

from documents import DOCUMENTS

st.set_page_config(
    page_title="Technical AI Model Laboratory",
    page_icon="🛠️",
    layout="wide",
)

# ---------- Readable dark theme ----------
BG, PANEL, BORDER = "#071321", "#12263B", "#29445F"
TEXT, MUTED = "#F0F6FF", "#C2D3E6"
BLUE, CYAN, ORANGE = "#22A7F0", "#38C8F4", "#FF9B45"
GREEN, RED, PURPLE, YELLOW = "#19D99A", "#FF5964", "#B78CFF", "#FFD84D"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{{font-family:Inter,Arial,sans-serif!important}}
.stApp{{background:radial-gradient(ellipse at 55% 0%,#102943 0%,{BG} 58%);color:{TEXT}}}
.block-container{{max-width:1800px;padding:1.1rem 1.35rem 2rem}}
[data-testid="stSidebar"]{{background:linear-gradient(180deg,#0D1B2B,#091625);border-right:1px solid {BORDER}}}
h1{{font-size:2.1rem!important;color:{TEXT}!important;font-weight:800!important}}
h2{{font-size:1.55rem!important;color:{TEXT}!important}}
h3{{font-size:1.12rem!important;color:{TEXT}!important}}
p,li,label,.stMarkdown{{color:{TEXT};font-size:15px;line-height:1.55}}
[data-testid="stCaptionContainer"] p{{color:{MUTED}!important;font-size:13px!important}}
.brand{{display:flex;justify-content:space-between;align-items:center;gap:18px;padding:2px 2px 15px;border-bottom:1px solid {BORDER};margin-bottom:15px}}
.brand-left{{display:flex;align-items:center;gap:17px}}
.brand-icon{{font-size:36px;background:#123454;border-radius:14px;padding:8px 14px}}
.brand-title{{font-size:29px;font-weight:800;color:#F4F8FF;line-height:1.2}}
.brand-sub{{font-size:17px;color:#C5D6EA;margin-top:4px}}
.online{{border:1px solid #00B987;background:#06352F;color:#45F0B7;padding:8px 13px;border-radius:9px;font-weight:700;font-size:13px;text-align:center}}
.micro{{font-size:12px;color:{MUTED};text-align:right;margin-top:6px}}
.section-title{{font-size:23px;font-weight:800;border-left:4px solid {BLUE};padding-left:13px;margin:6px 0}}
.section-sub{{font-size:15px;color:#C6D5E7;margin:0 0 16px 17px}}
.panel{{background:linear-gradient(145deg,#142A40,#0F2032);border:1px solid {BORDER};border-radius:13px;padding:15px 16px}}
.ptitle{{font-size:16px;font-weight:750;color:#F1F6FF;margin-bottom:12px}}
.takeaway{{background:linear-gradient(110deg,#211F43,#17243B);border:1px solid #51447B;border-radius:13px;padding:16px 18px}}
.take-title{{font-size:16px;font-weight:750;color:#D2B4FF;margin-bottom:8px}}
div[data-testid="stMetric"]{{background:{PANEL};border:1px solid {BORDER};border-radius:11px;padding:12px 15px}}
div[data-testid="stMetricLabel"]{{color:{MUTED}!important;font-size:13px!important}}
div[data-testid="stMetricValue"]{{color:{CYAN}!important;font-size:27px!important;font-weight:800!important}}
.stButton>button{{background:linear-gradient(90deg,#1267B7,#0B8CBF);color:white!important;border:1px solid #3DABEC;border-radius:9px;min-height:42px;font-weight:750!important}}
.stButton>button:hover{{background:#147DBE;border-color:#8BD9FF}}
.stTextInput input,.stTextArea textarea{{background:#0B1B2C!important;color:#F5F9FF!important;border:1px solid #3D5876!important;font-size:15px!important}}
div[data-baseweb="select"]>div{{background:#0B1B2C!important;color:#F5F9FF!important;border-color:#3D5876!important}}
div[data-testid="stSlider"] label{{color:#F0F6FF!important;font-size:14px!important}}
div[data-testid="stDataFrame"]{{border:1px solid {BORDER};border-radius:9px}}
hr{{border-color:{BORDER}}}
@media(max-width:900px){{.brand-title{{font-size:21px}}.brand-sub{{font-size:13px}}.online{{display:none}}.block-container{{padding:.8rem}}}}
</style>
""", unsafe_allow_html=True)

# ---------- Synthetic Naive Bayes training data ----------
TRAINING_DATA = [
    ("What causes packet loss and poor communication?", "troubleshooting"),
    ("How can I investigate an intermittent connection?", "troubleshooting"),
    ("Why is the communication link unstable?", "troubleshooting"),
    ("What causes degraded link quality?", "troubleshooting"),
    ("Why does the equipment restart repeatedly?", "fault_isolation"),
    ("How can I investigate unexpected restarts?", "fault_isolation"),
    ("Could unstable power cause a system reset?", "fault_isolation"),
    ("What can cause repeated equipment resets?", "fault_isolation"),
    ("How should I inspect equipment regularly?", "preventive_maintenance"),
    ("What are the routine inspection steps?", "preventive_maintenance"),
    ("How should operating temperature be monitored?", "preventive_maintenance"),
    ("What should a preventive inspection include?", "preventive_maintenance"),
]


# ---------- TF-IDF document retrieval ----------
@st.cache_resource
def build_retriever():
    texts = [d["title"] + " " + d["text"] for d in DOCUMENTS]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    return vectorizer, vectorizer.fit_transform(texts)


def retrieve(q, k=3):
    vectorizer, matrix = build_retriever()
    scores = cosine_similarity(
        vectorizer.transform([q]), matrix
    ).flatten()

    results = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] <= 0:
            continue
        doc = DOCUMENTS[i].copy()
        doc["score"] = float(scores[i])
        results.append(doc)
        if len(results) >= k:
            break
    return results


# ---------- Naive Bayes query classifier ----------
@st.cache_resource
def build_nb():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )),
        ("nb", MultinomialNB()),
    ])
    model.fit(
        [x for x, y in TRAINING_DATA],
        [y for x, y in TRAINING_DATA],
    )
    return model


def classify(q):
    model = build_nb()
    prediction = model.predict([q])[0]
    probabilities = model.predict_proba([q])[0]
    return prediction, dict(zip(model.classes_, probabilities))


def rerank(q, category, k=5):
    results = retrieve(q, len(DOCUMENTS))
    for doc in results:
        doc["category_bonus"] = (
            0.15 if doc["category"] == category else 0.0
        )
        doc["adjusted_score"] = (
            doc["score"] + doc["category_bonus"]
        )
    return sorted(
        results,
        key=lambda d: d["adjusted_score"],
        reverse=True,
    )[:k]


def show_results(results, adjusted=False):
    if not results:
        st.warning("No matching terms found. Try different wording.")
        return

    for i, doc in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {doc['title']}**")
            line = (
                f"Source: `{doc['id']}` · "
                f"Category: `{doc['category']}` · "
                f"Similarity: `{doc['score']:.3f}`"
            )
            if adjusted:
                line += (
                    f" · Bonus: `{doc['category_bonus']:.2f}`"
                    f" · Adjusted: `{doc['adjusted_score']:.3f}`"
                )
            st.caption(line)
            st.write(doc["text"])


# ---------- HMM and Viterbi algorithm ----------
STATES = ["Normal", "Degraded", "Fault"]
OBS = [
    "Normal temperature",
    "Elevated temperature",
    "High packet loss",
    "Repeated restart",
]

START = np.array([0.80, 0.15, 0.05])
TRANS = np.array([
    [0.80, 0.18, 0.02],
    [0.15, 0.65, 0.20],
    [0.05, 0.20, 0.75],
])
EMIT = np.array([
    [0.70, 0.10, 0.15, 0.05],
    [0.15, 0.45, 0.30, 0.10],
    [0.05, 0.15, 0.25, 0.55],
])


def viterbi(obs):
    if not obs:
        return [], 0.0

    indices = [OBS.index(x) for x in obs]
    dp = np.full((len(indices), 3), -np.inf)
    back = np.zeros((len(indices), 3), dtype=int)
    dp[0] = np.log(START) + np.log(EMIT[:, indices[0]])

    for t in range(1, len(indices)):
        for current in range(3):
            candidates = dp[t - 1] + np.log(TRANS[:, current])
            previous = int(np.argmax(candidates))
            dp[t, current] = (
                candidates[previous]
                + np.log(EMIT[current, indices[t]])
            )
            back[t, current] = previous

    last = int(np.argmax(dp[-1]))
    log_probability = float(dp[-1, last])
    path = [last]

    for t in range(len(indices) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))

    path.reverse()
    return [STATES[i] for i in path], log_probability


def style_plot(fig, height=280):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,29,.45)",
        font=dict(color="#E8F2FF", family="Inter,Arial", size=12),
        margin=dict(l=12, r=12, t=32, b=16),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="left", x=0, font=dict(size=10),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(132,164,198,.16)", linecolor="#536A83"
    )
    fig.update_yaxes(
        gridcolor="rgba(132,164,198,.16)", linecolor="#536A83"
    )
    return fig


# ---------- Branded header and sidebar ----------
st.markdown("""
<div class="brand">
  <div class="brand-left">
    <div class="brand-icon">🛠️</div>
    <div>
      <div class="brand-title">Technical AI Model Laboratory</div>
      <div class="brand-sub">VAE | Naive Bayes | HMM | Agentic RAG</div>
    </div>
  </div>
  <div>
    <div class="online">☁ &nbsp; Online Demo (Streamlit Cloud)</div>
    <div class="micro">Synthetic data only | For educational use</div>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⓘ About This Demo")
    st.write(
        "An interactive demonstration of VAE, Naive Bayes and HMM "
        "within an Agentic RAG concept for technical maintenance data."
    )
    st.divider()
    page = st.radio(
        "Navigate laboratory",
        [
            "VAE Laboratory",
            "Naive Bayes Laboratory",
            "HMM Laboratory",
            "Agentic RAG",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption(f"Corpus: {len(DOCUMENTS)} synthetic documents")
    st.caption("No external LLM/API is connected.")

# ============================================================
# VAE MODEL: TRAIN ON SYNTHETIC NORMAL TELEMETRY
# ============================================================

FEATURES = ["Temperature", "Vibration", "Voltage", "Packet Loss"]


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
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar


@st.cache_resource
def train_vae():
    # Synthetic normal telemetry only.
    rng = np.random.default_rng(42)
    n = 900

    temperature = rng.normal(55, 5, n)
    vibration = rng.normal(1.0, 0.22, n)
    voltage = rng.normal(28.0, 0.8, n)
    packet_loss = np.clip(rng.normal(0.6, 0.25, n), 0, None)

    normal_data = np.column_stack([
        temperature, vibration, voltage, packet_loss
    ])

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(normal_data).astype(np.float32)

    torch.manual_seed(42)
    model = TelemetryVAE(input_dim=4, latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
    x_tensor = torch.tensor(x_scaled)

    model.train()
    for _ in range(120):
        optimizer.zero_grad()

        reconstruction, mu, logvar = model(x_tensor)

        reconstruction_loss = nn.functional.mse_loss(
            reconstruction, x_tensor
        )

        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )

        loss = reconstruction_loss + 0.015 * kl_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        reconstruction, mu, logvar = model(x_tensor)
        errors = torch.mean(
            (reconstruction - x_tensor) ** 2, dim=1
        ).numpy()
        latent = mu.numpy()

    threshold = float(np.quantile(errors, 0.95))

    return model, scaler, threshold, normal_data, latent, errors


def vae_predict(values):
    model, scaler, threshold, _, _, _ = train_vae()

    scaled = scaler.transform(
        np.array(values, dtype=float).reshape(1, -1)
    ).astype(np.float32)

    tensor = torch.tensor(scaled)

    model.eval()
    with torch.no_grad():
        reconstruction, mu, logvar = model(tensor)

        error = float(torch.mean(
            (reconstruction - tensor) ** 2, dim=1
        ).item())

        reconstructed_original = scaler.inverse_transform(
            reconstruction.numpy()
        )[0]

        latent_point = mu.numpy()[0]

    return error, threshold, reconstructed_original, latent_point


# ============================================================
# VAE LABORATORY
# ============================================================

if page == "VAE Laboratory":
    st.markdown(
        '<div class="section-title">'
        'Variational Autoencoder (VAE) – Anomaly Detection</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">A VAE trained on synthetic normal '
        'telemetry. Adjust readings to inspect reconstruction error '
        'and how an anomaly signal could guide RAG retrieval.</div>',
        unsafe_allow_html=True,
    )

    model, scaler, threshold, normal_data, latent, training_errors = train_vae()

    a, b, c = st.columns([1.0, 1.28, 0.9], gap="medium")

    with a:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '☷ &nbsp; 1. Adjust Input Telemetry</div>',
            unsafe_allow_html=True,
        )

        temp = st.slider(
            "Temperature (°C)", 35, 85, 68, key="temp"
        )
        vib = st.slider(
            "Vibration (mm/s)", 0.2, 2.5, 2.1, 0.1, key="vib"
        )
        volt = st.slider(
            "Voltage (V)", 24.0, 32.0, 27.5, 0.1, key="volt"
        )
        packet = st.slider(
            "Packet Loss (%)", 0.0, 5.0, 1.2, 0.1, key="packet"
        )

        if st.button(
            "↻ Reset to Normal Values", use_container_width=True
        ):
            st.session_state.temp = 55
            st.session_state.vib = 1.0
            st.session_state.volt = 28.0
            st.session_state.packet = 0.5
            st.rerun()

        st.caption(
            "All data are synthetic; normal training patterns are "
            "generated in code."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    current = np.array([temp, vib, volt, packet], dtype=float)
    score, threshold, reconstructed, latent_point = vae_predict(current)
    is_anomaly = score >= threshold

    with b:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '📈 &nbsp; 2. Input vs. VAE Reconstruction</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_bar(
            x=FEATURES, y=current,
            name="Actual Input", marker_color=BLUE,
        )
        fig.add_bar(
            x=FEATURES, y=reconstructed,
            name="VAE Reconstructed", marker_color=ORANGE,
        )
        fig.update_layout(barmode="group")

        st.plotly_chart(
            style_plot(fig, 280),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Reconstruction is produced by the trained neural network."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with c:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '◉ &nbsp; 3. Anomaly Score</div>',
            unsafe_allow_html=True,
        )

        gauge_value = min(score / max(threshold, 1e-8), 2.0)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={
                "font": {"size": 32, "color": TEXT},
                "valueformat": ".2f",
                "suffix": "×",
            },
            title={
                "text": "Error / threshold",
                "font": {"size": 13, "color": MUTED},
            },
            gauge={
                "axis": {"range": [0, 2], "tickcolor": MUTED},
                "bar": {
                    "color": RED if is_anomaly else GREEN,
                    "thickness": 0.22,
                },
                "bgcolor": PANEL,
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 1], "color": "#087A62"},
                    {"range": [1, 1.5], "color": "#A36A1B"},
                    {"range": [1.5, 2], "color": "#7E2638"},
                ],
                "threshold": {
                    "line": {"color": YELLOW, "width": 3},
                    "thickness": 0.85,
                    "value": 1.0,
                },
            },
        ))

        gauge.update_layout(
            height=238,
            margin=dict(l=8, r=8, t=35, b=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT),
        )

        st.plotly_chart(
            gauge,
            use_container_width=True,
            config={"displayModeBar": False},
        )

        st.caption(
            f"Reconstruction MSE: {score:.4f} · "
            f"95th-percentile training threshold: {threshold:.4f}"
        )

        if is_anomaly:
            st.markdown(
                '<div style="background:#3D1B27;border:1px solid #A83A54;'
                'border-radius:10px;padding:12px">'
                '<b style="color:#FFDDE4">● &nbsp; Anomaly signal</b><br>'
                'Reconstruction error is above the synthetic training '
                'threshold.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#063D36;border:1px solid #079A78;'
                'border-radius:10px;padding:12px">'
                '<b style="color:#D5FFF2">● &nbsp; Normal-like input</b><br>'
                'Reconstruction error is within the synthetic training '
                'threshold.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    d, e = st.columns([1, 1.1], gap="medium")

    with d:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '♧ &nbsp; 4. Learned Latent Space</div>',
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_scatter(
            x=latent[:, 0], y=latent[:, 1],
            mode="markers",
            name="Synthetic normal training data",
            marker=dict(size=5, color=BLUE, opacity=0.55),
        )
        fig.add_scatter(
            x=[latent_point[0]], y=[latent_point[1]],
            mode="markers", name="Current input",
            marker=dict(
                size=15, color=YELLOW, symbol="star",
                line=dict(width=1, color="#FFFFFF"),
            ),
        )
        fig.update_layout(
            xaxis_title="Latent mean z₁",
            yaxis_title="Latent mean z₂",
        )

        st.plotly_chart(
            style_plot(fig, 275),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Latent coordinates are encoder means from the trained VAE."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with e:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '⌁ &nbsp; Reconstruction Error Distribution</div>',
            unsafe_allow_html=True,
        )

        hist = go.Figure()
        hist.add_histogram(
            x=training_errors,
            name="Normal training error",
            marker_color=BLUE,
            opacity=0.8,
        )
        hist.add_vline(
            x=threshold, line_dash="dash",
            line_color=YELLOW,
            annotation_text="95th percentile",
        )
        hist.add_vline(
            x=score, line_color=RED,
            annotation_text="Current input",
        )
        hist.update_layout(
            xaxis_title="Reconstruction MSE",
            yaxis_title="Count",
            bargap=0.05,
        )

        st.plotly_chart(
            style_plot(hist, 275),
            use_container_width=True,
            config={"displayModeBar": False},
        )
        st.caption(
            "Threshold is derived from synthetic training errors, "
            "not real-world validation."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="takeaway">
      <div class="take-title">✦ &nbsp; Key Takeaways</div>
      <div style="display:grid;grid-template-columns:1.15fr 1fr;gap:22px">
        <div>
          • The VAE learns a latent representation of synthetic normal telemetry.<br>
          • Reconstruction MSE measures how poorly a sample is reconstructed.<br>
          • The threshold is set at the 95th percentile of training errors.
        </div>
        <div style="border-left:1px solid #62558B;padding-left:20px">
          <div class="take-title">⚙ &nbsp; Use Cases in Technical RAG</div>
          • Flag unusual telemetry before retrieval.<br>
          • Use the signal to prioritize relevant maintenance documents.<br>
          • Validate thresholds on representative data before practical use.
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(
        "Use VAE signal to retrieve maintenance evidence",
        type="primary",
        use_container_width=True,
    ):
        query = (
            "equipment temperature vibration voltage packet loss "
            "anomaly troubleshooting fault"
        )
        category = (
            "fault_isolation" if is_anomaly
            else "preventive_maintenance"
        )
        results = rerank(query, category, 3)
        st.subheader("Retrieved evidence based on anomaly signal")
        show_results(results, True)
        st.caption(
            "This is a demonstration trigger, not a validated diagnosis."
        )

# ============================================================
# NAIVE BAYES LABORATORY
# ============================================================

elif page == "Naive Bayes Laboratory":
    st.markdown(
        '<div class="section-title">'
        'Naive Bayes – Query Intent and Retrieval Routing</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Classify a question and inspect how '
        'its predicted category adds context to retrieval.</div>',
        unsafe_allow_html=True,
    )

    q = st.text_input(
        "Technical maintenance question",
        "Why does the equipment repeatedly restart?",
        key="nb_question",
    )

    if st.button(
        "CLASSIFY QUERY AND RETRIEVE",
        type="primary",
        use_container_width=True,
    ):
        pred, probs = classify(q)

        x, y = st.columns([0.9, 1.2], gap="medium")

        with x:
            st.markdown(
                '<div class="panel"><div class="ptitle">'
                '🧭 Predicted query category</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"## {pred.replace('_', ' ').title()}")

            df = pd.DataFrame({
                "Category": list(probs.keys()),
                "Probability": list(probs.values()),
            }).sort_values("Probability")

            fig = px.bar(
                df,
                x="Probability",
                y="Category",
                orientation="h",
                range_x=[0, 1],
                color_discrete_sequence=[CYAN],
            )
            st.plotly_chart(
                style_plot(fig, 270),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Small synthetic training set; probabilities are "
                "not calibrated confidence estimates."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with y:
            st.markdown(
                '<div class="panel"><div class="ptitle">'
                '⌕ Retrieval results</div>',
                unsafe_allow_html=True,
            )
            st.markdown("**Baseline retrieval**")
            show_results(retrieve(q, 2))

            st.markdown("**Category-aware reranking**")
            show_results(rerank(q, pred, 3), True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.info(
            "The fixed category bonus is a heuristic. It can improve "
            "or worsen retrieval and must be evaluated."
        )
    else:
        st.markdown(
            '<div class="panel"><div class="ptitle">'
            'How it connects to RAG</div>'
            'Naive Bayes predicts whether a question concerns '
            'troubleshooting, fault isolation or preventive maintenance. '
            'That intent can help filter or prioritize candidate passages.'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# HMM LABORATORY
# ============================================================

elif page == "HMM Laboratory":
    st.markdown(
        '<div class="section-title">'
        'Hidden Markov Model – Temporal State Tracking</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Infer the most likely hidden-state '
        'sequence from ordered observations using Viterbi decoding.</div>',
        unsafe_allow_html=True,
    )

    st.warning(
        "Transition and emission probabilities are illustrative "
        "assumptions, not trained equipment parameters."
    )

    obs = st.multiselect(
        "Select observations in chronological order",
        OBS,
        default=OBS,
        key="hmm_obs",
    )

    if obs:
        states, logp = viterbi(obs)
        x, y = st.columns([1.1, 1], gap="medium")

        with x:
            st.markdown(
                '<div class="panel"><div class="ptitle">'
                '⌁ Inferred sequence</div>',
                unsafe_allow_html=True,
            )

            st.dataframe(
                pd.DataFrame({
                    "Step": range(1, len(obs) + 1),
                    "Observation": obs,
                    "Inferred state": states,
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### " + " → ".join(states))
            st.metric("Final inferred state", states[-1])
            st.caption(
                f"Best-path log probability: {logp:.3f}; "
                "this is not a fault probability."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with y:
            st.markdown(
                '<div class="panel"><div class="ptitle">'
                '⇢ State transition matrix</div>',
                unsafe_allow_html=True,
            )

            fig = px.imshow(
                pd.DataFrame(TRANS, index=STATES, columns=STATES),
                text_auto=".2f",
                color_continuous_scale="Blues",
            )
            st.plotly_chart(
                style_plot(fig, 300),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        category = (
            "fault_isolation"
            if states[-1] == "Fault"
            else (
                "troubleshooting"
                if states[-1] == "Degraded"
                else "preventive_maintenance"
            )
        )

        st.markdown(
            '<div class="panel"><div class="ptitle">'
            '⌕ Use inferred state as RAG context</div>',
            unsafe_allow_html=True,
        )
        st.write(
            "Proposed retrieval category: **"
            + category.replace("_", " ").title()
            + "**"
        )

        if st.button(
            "RETRIEVE STATE-RELEVANT DOCUMENTS",
            type="primary",
            use_container_width=True,
        ):
            show_results(rerank(" ".join(obs), category, 3), True)

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# AGENTIC RAG LABORATORY
# ============================================================

elif page == "Agentic RAG":
    st.markdown(
        '<div class="section-title">'
        'Agentic RAG – Evidence-Grounded Technical Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">A transparent workflow: '
        'query interpretation → retrieval → evidence inspection.</div>',
        unsafe_allow_html=True,
    )

    q = st.text_input(
        "Ask the technical assistant",
        placeholder=(
            "Why might the unit restart while packet loss is increasing?"
        ),
        key="agent_query",
    )

    x, y = st.columns(2)

    with x:
        k = st.selectbox(
            "Evidence passages",
            [1, 2, 3, 5],
            index=2,
        )

    with y:
        use_nb = st.toggle(
            "Use Naive Bayes query intent",
            value=True,
        )

    if st.button(
        "RUN RETRIEVAL WORKFLOW",
        type="primary",
        use_container_width=True,
    ):
        if not q.strip():
            st.warning("Enter a question first.")
        else:
            if use_nb:
                category, probs = classify(q)
            else:
                category, probs = None, {}

            x, y = st.columns([0.9, 1.2], gap="medium")

            with x:
                st.markdown(
                    '<div class="panel"><div class="ptitle">'
                    '01 · Query interpretation</div>',
                    unsafe_allow_html=True,
                )

                if category:
                    st.markdown(
                        "**Predicted intent:** "
                        + category.replace("_", " ").title()
                    )

                    st.dataframe(
                        pd.DataFrame({
                            "Category": list(probs.keys()),
                            "Probability": list(probs.values()),
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.write("Intent classification disabled.")

                st.success("Synthetic corpus ready")
                st.success("Retrieval executed")
                st.markdown("</div>", unsafe_allow_html=True)

            with y:
                st.markdown(
                    '<div class="panel"><div class="ptitle">'
                    '02 · Retrieved evidence</div>',
                    unsafe_allow_html=True,
                )

                if category:
                    show_results(rerank(q, category, k), True)
                else:
                    show_results(retrieve(q, k))

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                '<div class="takeaway">'
                '<div class="take-title">'
                '03 · Evidence-grounded response status</div>'
                'This prototype retrieves passages and exposes decision '
                'signals. It does not connect an LLM or autonomously '
                'execute maintenance actions. A production system should '
                'cite approved sources and defer operational decisions '
                'to authorized procedures and qualified personnel.'
                '</div>',
                unsafe_allow_html=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="micro" style="text-align:center;padding:18px 0">'
    'Technical AI Model Laboratory · Synthetic educational demonstrator · '
    'Not for operational maintenance decisions</div>',
    unsafe_allow_html=True,
)
