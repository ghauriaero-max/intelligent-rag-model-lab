import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from documents import DOCUMENTS

st.set_page_config(page_title="Technical AI Model Laboratory", page_icon="🛠️", layout="wide")

# ---- Visual system: high-contrast dark engineering dashboard ----
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

# ---- Synthetic training examples, retrieval and classifier ----
TRAINING_DATA = [
("What causes packet loss and poor communication?","troubleshooting"),
("How can I investigate an intermittent connection?","troubleshooting"),
("Why is the communication link unstable?","troubleshooting"),
("What causes degraded link quality?","troubleshooting"),
("Why does the equipment restart repeatedly?","fault_isolation"),
("How can I investigate unexpected restarts?","fault_isolation"),
("Could unstable power cause a system reset?","fault_isolation"),
("What can cause repeated equipment resets?","fault_isolation"),
("How should I inspect equipment regularly?","preventive_maintenance"),
("What are the routine inspection steps?","preventive_maintenance"),
("How should operating temperature be monitored?","preventive_maintenance"),
("What should a preventive inspection include?","preventive_maintenance"),
]

@st.cache_resource
def build_retriever():
    texts = [d["title"] + " " + d["text"] for d in DOCUMENTS]
    vec = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2), sublinear_tf=True)
    return vec, vec.fit_transform(texts)

def retrieve(q, k=3):
    vec, mat = build_retriever()
    scores = cosine_similarity(vec.transform([q]), mat).flatten()
    out = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] <= 0: continue
        d = DOCUMENTS[i].copy()
        d["score"] = float(scores[i])
        out.append(d)
        if len(out) >= k: break
    return out

@st.cache_resource
def build_nb():
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1,2))),
        ("nb", MultinomialNB()),
    ])
    model.fit([x for x,y in TRAINING_DATA], [y for x,y in TRAINING_DATA])
    return model

def classify(q):
    model = build_nb()
    pred = model.predict([q])[0]
    probs = model.predict_proba([q])[0]
    return pred, dict(zip(model.classes_, probs))

def rerank(q, category, k=5):
    results = retrieve(q, len(DOCUMENTS))
    for d in results:
        d["category_bonus"] = 0.15 if d["category"] == category else 0.0
        d["adjusted_score"] = d["score"] + d["category_bonus"]
    return sorted(results, key=lambda d:d["adjusted_score"], reverse=True)[:k]

def show_results(results, adjusted=False):
    if not results:
        st.warning("No matching terms found. Try different wording.")
        return
    for i,d in enumerate(results,1):
        with st.container(border=True):
            st.markdown(f"**{i}. {d['title']}**")
            line = f"Source: `{d['id']}` · Category: `{d['category']}` · Similarity: `{d['score']:.3f}`"
            if adjusted:
                line += f" · Bonus: `{d['category_bonus']:.2f}` · Adjusted: `{d['adjusted_score']:.3f}`"
            st.caption(line)
            st.write(d["text"])

# ---- HMM / Viterbi ----
STATES = ["Normal","Degraded","Fault"]
OBS = ["Normal temperature","Elevated temperature","High packet loss","Repeated restart"]
START = np.array([.80,.15,.05])
TRANS = np.array([[.80,.18,.02],[.15,.65,.20],[.05,.20,.75]])
EMIT = np.array([[.70,.10,.15,.05],[.15,.45,.30,.10],[.05,.15,.25,.55]])

def viterbi(obs):
    if not obs: return [], 0.0
    oi = [OBS.index(x) for x in obs]
    dp = np.full((len(oi),3), -np.inf)
    back = np.zeros((len(oi),3),dtype=int)
    dp[0] = np.log(START)+np.log(EMIT[:,oi[0]])
    for t in range(1,len(oi)):
        for cur in range(3):
            candidates = dp[t-1]+np.log(TRANS[:,cur])
            prev = int(np.argmax(candidates))
            dp[t,cur] = candidates[prev]+np.log(EMIT[cur,oi[t]])
            back[t,cur] = prev
    last = int(np.argmax(dp[-1]))
    logp = float(dp[-1,last])
    path = [last]
    for t in range(len(oi)-1,0,-1): path.append(int(back[t,path[-1]]))
    path.reverse()
    return [STATES[i] for i in path], logp

def style_plot(fig, height=280):
    fig.update_layout(
        height=height,paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(5,16,29,.45)",
        font=dict(color="#E8F2FF",family="Inter,Arial",size=12),
        margin=dict(l=12,r=12,t=32,b=16),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0,font=dict(size=10)),
    )
    fig.update_xaxes(gridcolor="rgba(132,164,198,.16)",linecolor="#536A83")
    fig.update_yaxes(gridcolor="rgba(132,164,198,.16)",linecolor="#536A83")
    return fig

# ---- Header ----
st.markdown("""
<div class="brand">
  <div class="brand-left">
    <div class="brand-icon">🛠️</div>
    <div><div class="brand-title">Technical AI Model Laboratory</div>
    <div class="brand-sub">VAE | Naive Bayes | HMM | Agentic RAG</div></div>
  </div>
  <div><div class="online">☁ &nbsp; Online Demo (Streamlit Cloud)</div>
  <div class="micro">Synthetic data only | For educational use</div></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⓘ About This Demo")
    st.write("An interactive demonstration of VAE, Naive Bayes and HMM within an Agentic RAG concept for technical maintenance data.")
    st.divider()
    page = st.radio("Navigate laboratory",[
        "VAE Laboratory","Naive Bayes Laboratory","HMM Laboratory","Agentic RAG"
    ],label_visibility="collapsed")
    st.divider()
    st.caption(f"Corpus: {len(DOCUMENTS)} synthetic documents")
    st.caption("No external LLM/API is connected.")

# ============================================================
# VAE VISUAL LAB (dashboard-inspired; score is illustrative proxy)
# ============================================================
if page == "VAE Laboratory":
    st.markdown('<div class="section-title">Variational Autoencoder (VAE) – Anomaly Detection</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore how a VAE may learn normal telemetry patterns and flag unusual observations using reconstruction error.</div>',unsafe_allow_html=True)

    a,b,c = st.columns([1.0,1.28,.9],gap="medium")
    with a:
        st.markdown('<div class="panel"><div class="ptitle">☷ &nbsp; 1. Adjust Input Telemetry</div>',unsafe_allow_html=True)
        temp = st.slider("Temperature (°C)",20,100,68,key="temp")
        vib = st.slider("Vibration (mm/s)",0.0,5.0,2.1,.1,key="vib")
        volt = st.slider("Voltage (V)",18.0,32.0,27.5,.1,key="volt")
        packet = st.slider("Packet Loss (%)",0.0,20.0,1.2,.1,key="packet")
        if st.button("↻ Reset to Normal Values",use_container_width=True):
            st.session_state.temp=55
            st.session_state.vib=1.0
            st.session_state.volt=28.0
            st.session_state.packet=.5
            st.rerun()
        st.caption("Synthetic telemetry controls.")
        st.markdown('</div>',unsafe_allow_html=True)

    normal=np.array([55.,1.,28.,.5])
    current=np.array([temp,vib,volt,packet])
    scales=np.array([25.,1.5,4.,5.])
    delta=(current-normal)/scales
    score=float(np.sqrt(np.mean(delta**2)))
    reconstructed=normal+.72*(current-normal)
    threshold=.30

    with b:
        st.markdown('<div class="panel"><div class="ptitle">📈 &nbsp; 2. Reconstruction Comparison</div>',unsafe_allow_html=True)
        labels=["Temperature","Vibration","Voltage","Packet Loss"]
        fig=go.Figure()
        fig.add_bar(x=labels,y=current,name="Actual Input",marker_color=BLUE)
        fig.add_bar(x=labels,y=reconstructed,name="VAE Reconstructed",marker_color=ORANGE)
        fig.update_layout(barmode="group")
        st.plotly_chart(style_plot(fig,280),use_container_width=True,config={"displayModeBar":False})
        st.caption("Illustrative reconstruction, not output from a trained VAE.")
        st.markdown('</div>',unsafe_allow_html=True)

    with c:
        st.markdown('<div class="panel"><div class="ptitle">◉ &nbsp; 3. Anomaly Score</div>',unsafe_allow_html=True)
        gauge=go.Figure(go.Indicator(
            mode="gauge+number",value=min(score,1.0),
            number={"font":{"size":32,"color":TEXT},"valueformat":".2f"},
            title={"text":"Illustrative error proxy","font":{"size":13,"color":MUTED}},
            gauge={"axis":{"range":[0,1],"tickcolor":MUTED},
                   "bar":{"color":BLUE,"thickness":.22},"bgcolor":PANEL,"borderwidth":0,
                   "steps":[{"range":[0,.30],"color":"#087A62"},{"range":[.30,.60],"color":"#A36A1B"},{"range":[.60,1],"color":"#7E2638"}],
                   "threshold":{"line":{"color":YELLOW,"width":3},"thickness":.85,"value":threshold}}
        ))
        gauge.update_layout(height=238,margin=dict(l=8,r=8,t=35,b=5),paper_bgcolor="rgba(0,0,0,0)",font=dict(color=TEXT))
        st.plotly_chart(gauge,use_container_width=True,config={"displayModeBar":False})
        if score>=threshold:
            st.markdown('<div style="background:#3D1B27;border:1px solid #A83A54;border-radius:10px;padding:12px"><b style="color:#FFDDE4">● &nbsp; Anomaly signal</b><br>Proxy score exceeds demo threshold.</div>',unsafe_allow_html=True)
        else:
            st.markdown('<div style="background:#063D36;border:1px solid #079A78;border-radius:10px;padding:12px"><b style="color:#D5FFF2">● &nbsp; Normal</b><br>Proxy score is below demo threshold.</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

    st.write("")
    d,e=st.columns([1,1.1],gap="medium")
    with d:
        st.markdown('<div class="panel"><div class="ptitle">♧ &nbsp; 4. Latent Space Visualization</div>',unsafe_allow_html=True)
        rng=np.random.default_rng(14)
        fig=go.Figure()
        fig.add_scatter(x=rng.normal(0,.85,350),y=rng.normal(0,.85,350),mode="markers",name="Normal Data",marker=dict(size=4,color=BLUE,opacity=.75))
        fig.add_scatter(x=[2,2.6,2.4,3,2.8],y=[.9,1.2,.2,.7,-.1],mode="markers",name="Anomalies",marker=dict(size=9,color=RED))
        fig.add_scatter(x=[float(np.clip(delta[0],-3.5,3.5))],y=[float(np.clip(delta[1],-3.5,3.5))],mode="markers",name="Current Input",marker=dict(size=15,color=YELLOW,symbol="star"))
        fig.update_layout(xaxis_title="Latent Dimension 1",yaxis_title="Latent Dimension 2")
        fig.update_xaxes(range=[-4,4]); fig.update_yaxes(range=[-4,4])
        st.plotly_chart(style_plot(fig,275),use_container_width=True,config={"displayModeBar":False})
        st.caption("Synthetic visualization; these are not learned latent embeddings.")
        st.markdown('</div>',unsafe_allow_html=True)
    with e:
        st.markdown('<div class="panel"><div class="ptitle">⌁ &nbsp; 5. Telemetry Time Series (Synthetic)</div>',unsafe_allow_html=True)
        rng=np.random.default_rng(27); t=np.arange(100)
        traces=[
            ("Temperature",.62+.10*np.sin(t/5)+rng.normal(0,.055,100),BLUE),
            ("Vibration",.22+.045*np.sin(t/6+1)+rng.normal(0,.018,100),ORANGE),
            ("Voltage",.42+.045*np.sin(t/11+2)+rng.normal(0,.015,100),GREEN),
            ("Packet Loss",.10+.03*np.sin(t/4+1)+rng.normal(0,.02,100),RED),
        ]
        fig=go.Figure()
        for name,vals,color in traces: fig.add_scatter(x=t,y=vals,name=name,line=dict(color=color,width=2.2))
        fig.update_layout(xaxis_title="Time Step",yaxis_title="Normalized Value")
        fig.update_yaxes(range=[0,1])
        st.plotly_chart(style_plot(fig,275),use_container_width=True,config={"displayModeBar":False})
        st.caption("Generated data for presentation; not operational telemetry.")
        st.markdown('</div>',unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="takeaway">
      <div class="take-title">✦ &nbsp; Key Takeaways</div>
      <div style="display:grid;grid-template-columns:1.15fr 1fr;gap:22px">
        <div>• A VAE learns a compressed representation of normal behavior.<br>• Reconstruction error measures how well input matches learned patterns.<br>• High error can flag unusual behavior, but does not prove a fault.</div>
        <div style="border-left:1px solid #62558B;padding-left:20px"><div class="take-title">⚙ &nbsp; Use Cases in Technical RAG</div>• Flag unusual telemetry before retrieval.<br>• Prioritize relevant maintenance documents.<br>• Combine with HMM and Naive Bayes for context.</div>
      </div>
    </div>
    """,unsafe_allow_html=True)
    st.warning("Important: the dashboard is VAE-inspired, but the score, reconstruction and latent plot are synthetic demonstrations, not outputs from a trained VAE.")

# ============================================================
# NAIVE BAYES
# ============================================================
elif page == "Naive Bayes Laboratory":
    st.markdown('<div class="section-title">Naive Bayes – Query Intent and Retrieval Routing</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Classify a question and inspect how its predicted category adds context to retrieval.</div>',unsafe_allow_html=True)
    q=st.text_input("Technical maintenance question","Why does the equipment repeatedly restart?",key="nb_question")
    if st.button("CLASSIFY QUERY AND RETRIEVE",type="primary",use_container_width=True):
        pred,probs=classify(q)
        x,y=st.columns([.9,1.2],gap="medium")
        with x:
            st.markdown('<div class="panel"><div class="ptitle">🧭 Predicted query category</div>',unsafe_allow_html=True)
            st.markdown(f"## {pred.replace('_',' ').title()}")
            df=pd.DataFrame({"Category":list(probs.keys()),"Probability":list(probs.values())}).sort_values("Probability")
            fig=px.bar(df,x="Probability",y="Category",orientation="h",range_x=[0,1],color_discrete_sequence=[CYAN])
            st.plotly_chart(style_plot(fig,270),use_container_width=True,config={"displayModeBar":False})
            st.caption("Small synthetic training set; probabilities are not calibrated.")
            st.markdown('</div>',unsafe_allow_html=True)
        with y:
            st.markdown('<div class="panel"><div class="ptitle">⌕ Retrieval results</div>',unsafe_allow_html=True)
            st.markdown("**Baseline**")
            show_results(retrieve(q,2))
            st.markdown("**Category-aware reranking**")
            show_results(rerank(q,pred,3),True)
            st.markdown('</div>',unsafe_allow_html=True)
        st.info("The fixed category bonus is a heuristic. It can improve or worsen retrieval and must be evaluated.")
    else:
        st.markdown('<div class="panel"><div class="ptitle">How it connects to RAG</div>Naive Bayes predicts whether a question concerns troubleshooting, fault isolation or preventive maintenance. That intent can help filter or prioritize candidate passages.</div>',unsafe_allow_html=True)

# ============================================================
# HMM
# ============================================================
elif page == "HMM Laboratory":
    st.markdown('<div class="section-title">Hidden Markov Model – Temporal State Tracking</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Infer the most likely hidden-state sequence from ordered observations using Viterbi decoding.</div>',unsafe_allow_html=True)
    st.warning("Transition and emission probabilities are illustrative assumptions, not trained equipment parameters.")
    obs=st.multiselect("Select observations in chronological order",OBS,default=OBS,key="hmm_obs")
    if obs:
        states,logp=viterbi(obs)
        x,y=st.columns([1.1,1],gap="medium")
        with x:
            st.markdown('<div class="panel"><div class="ptitle">⌁ Inferred sequence</div>',unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"Step":range(1,len(obs)+1),"Observation":obs,"Inferred state":states}),use_container_width=True,hide_index=True)
            st.markdown("#### "+" → ".join(states))
            st.metric("Final inferred state",states[-1])
            st.caption(f"Best-path log probability: {logp:.3f}; not a fault probability.")
            st.markdown('</div>',unsafe_allow_html=True)
        with y:
            st.markdown('<div class="panel"><div class="ptitle">⇢ State transition matrix</div>',unsafe_allow_html=True)
            fig=px.imshow(pd.DataFrame(TRANS,index=STATES,columns=STATES),text_auto=".2f",color_continuous_scale="Blues")
            st.plotly_chart(style_plot(fig,300),use_container_width=True,config={"displayModeBar":False})
            st.markdown('</div>',unsafe_allow_html=True)
        category="fault_isolation" if states[-1]=="Fault" else ("troubleshooting" if states[-1]=="Degraded" else "preventive_maintenance")
        st.markdown('<div class="panel"><div class="ptitle">⌕ Use inferred state as RAG context</div>',unsafe_allow_html=True)
        st.write("Proposed retrieval category: **"+category.replace("_"," ").title()+"**")
        if st.button("RETRIEVE STATE-RELEVANT DOCUMENTS",type="primary",use_container_width=True):
            show_results(rerank(" ".join(obs),category,3),True)
        st.markdown('</div>',unsafe_allow_html=True)

# ============================================================
# AGENTIC RAG
# ============================================================
elif page == "Agentic RAG":
    st.markdown('<div class="section-title">Agentic RAG – Evidence-Grounded Technical Assistant</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">A transparent workflow: query interpretation → retrieval → evidence inspection.</div>',unsafe_allow_html=True)
    q=st.text_input("Ask the technical assistant",placeholder="Why might the unit restart while packet loss is increasing?",key="agent_query")
    x,y=st.columns(2)
    with x: k=st.selectbox("Evidence passages",[1,2,3,5],index=2)
    with y: use_nb=st.toggle("Use Naive Bayes query intent",value=True)
    if st.button("RUN RETRIEVAL WORKFLOW",type="primary",use_container_width=True):
        if not q.strip(): st.warning("Enter a question first.")
        else:
            category,probs=classify(q) if use_nb else (None,{})
            x,y=st.columns([.9,1.2],gap="medium")
            with x:
                st.markdown('<div class="panel"><div class="ptitle">01 · Query interpretation</div>',unsafe_allow_html=True)
                if category:
                    st.markdown(f"**Predicted intent:** {category.replace('_',' ').title()}")
                    st.dataframe(pd.DataFrame({"Category":list(probs.keys()),"Probability":list(probs.values())}),use_container_width=True,hide_index=True)
                else: st.write("Intent classification disabled.")
                st.success("Synthetic corpus ready")
                st.success("Retrieval executed")
                st.markdown('</div>',unsafe_allow_html=True)
            with y:
                st.markdown('<div class="panel"><div class="ptitle">02 · Retrieved evidence</div>',unsafe_allow_html=True)
                show_results(rerank(q,category,k),True) if category else show_results(retrieve(q,k))
                st.markdown('</div>',unsafe_allow_html=True)
            st.markdown('<div class="takeaway"><div class="take-title">03 · Evidence-grounded response status</div>This prototype retrieves passages and exposes decision signals. It does not connect an LLM or autonomously execute maintenance actions. A production system should cite approved sources and defer operational decisions to authorized procedures and qualified personnel.</div>',unsafe_allow_html=True)

st.markdown('<div class="micro" style="text-align:center;padding:18px 0">Technical AI Model Laboratory · Synthetic educational demonstrator · Not for operational maintenance decisions</div>',unsafe_allow_html=True)
