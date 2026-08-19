import streamlit as st
from pipeline import run_research_pipeline  # noqa: F401  (kept for reference / CLI parity)
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

st.set_page_config(page_title="ResearchMind — AI Research Agent", page_icon="🧠", layout="wide")

# ----------------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------------
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""
if "current_step" not in st.session_state:
    st.session_state.current_step = -1  # -1 = idle, 0-3 = running, 4 = done
if "result" not in st.session_state:
    st.session_state.result = None

SUGGESTIONS = ["LLM agents 2025", "Fusion energy progress", "Future of remote work"]

STEPS = [
    ("01", "Search Agent", "Gathers recent web information"),
    ("02", "Reader Agent", "Scrapes & extracts deep content"),
    ("03", "Writer", "Drafts a structured report"),
    ("04", "Critic", "Scores and reviews the draft"),
]

# ----------------------------------------------------------------------------
# STYLE
# ----------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

#MainMenu, footer, header { visibility: hidden; }
.stApp {
    background: radial-gradient(circle at 20% 0%, #14131c 0%, #0a0a0f 45%, #08080b 100%);
}
.block-container { padding-top: 3rem; max-width: 1200px; }

/* Hero */
.eyebrow {
    color: #ff7a1a;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    letter-spacing: 0.35em;
    font-size: 0.75rem;
    text-align: center;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 5rem;
    line-height: 1;
    text-align: center;
    letter-spacing: -0.02em;
    margin: 0 0 1.4rem 0;
    color: #f5f4f2;
}
.hero-title .accent {
    background: linear-gradient(90deg, #ff9a3d, #ff5e1a);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero-sub {
    color: #9a97a3;
    font-size: 1.05rem;
    text-align: center;
    max-width: 640px;
    margin: 0 auto 3rem auto;
    line-height: 1.6;
}

/* Section labels */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    color: #ff7a1a;
    margin-bottom: 0.5rem;
    margin-top: 0.4rem;
}
.pipeline-heading {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.35rem;
    font-weight: 600;
    color: #f5f4f2;
    margin-bottom: 1rem;
}

/* Text input */
div[data-testid="stTextInput"] input {
    background: #141319 !important;
    border: 1px solid #2a2833 !important;
    border-radius: 10px !important;
    color: #f5f4f2 !important;
    padding: 0.85rem 1rem !important;
    font-size: 0.95rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border: 1px solid #ff7a1a !important;
    box-shadow: 0 0 0 1px #ff7a1a33 !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #6b6875 !important; }

/* Run button */
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #ff9a3d, #ff5e1a) !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600 !important;
    padding: 0.75rem 1rem !important;
    width: 100%;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    box-shadow: 0 8px 24px -8px #ff5e1a66;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 12px 28px -8px #ff5e1a99;
}
div[data-testid="stButton"] button:disabled {
    background: #232129 !important;
    color: #6b6875 !important;
    box-shadow: none;
}

/* Suggestion chips (secondary buttons) */
button[kind="secondary"] {
    background: #14131a !important;
    border: 1px solid #2a2833 !important;
    color: #b3b0bb !important;
    border-radius: 999px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.35rem 0.9rem !important;
    box-shadow: none !important;
}
button[kind="secondary"]:hover {
    border-color: #ff7a1a !important;
    color: #ff9a3d !important;
    transform: none !important;
}

/* Pipeline cards */
.pcard {
    background: #121118;
    border: 1px solid #232129;
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.85rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    transition: border-color 0.2s ease;
}
.pcard.running { border-color: #ff7a1a66; background: #17141a; }
.pcard.done { border-color: #2fae6633; }
.pcard-num {
    font-family: 'Space Grotesk', sans-serif;
    color: #ff7a1a;
    font-size: 0.75rem;
    font-weight: 700;
    margin-right: 0.6rem;
}
.pcard-title {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    color: #f5f4f2;
    font-size: 1rem;
}
.pcard-desc { color: #7f7c88; font-size: 0.82rem; margin-top: 0.2rem; }
.badge {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    white-space: nowrap;
    margin-top: 0.15rem;
}
.badge.waiting { background: #1e1c24; color: #6b6875; }
.badge.running { background: #ff7a1a22; color: #ff9a3d; animation: pulse 1.4s ease-in-out infinite; }
.badge.done { background: #2fae6622; color: #4fd88c; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.45; } }

/* Result tabs area */
.result-wrap { margin-top: 2.5rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------------
st.markdown('<div class="eyebrow">MULTI-AGENT AI SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Research<span class="accent">Mind</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Four specialized AI agents collaborate — searching, scraping, writing, '
    'and critiquing — to deliver a polished research report on any topic.</p>',
    unsafe_allow_html=True,
)


def render_pipeline_cards(current_step: int) -> str:
    """current_step: -1 idle, 0-3 that step is running, 4 = all done"""
    html = []
    for i, (num, title, desc) in enumerate(STEPS):
        if current_step == -1:
            state, label = "waiting", "WAITING"
        elif current_step == 4 or i < current_step:
            state, label = "done", "DONE"
        elif i == current_step:
            state, label = "running", "RUNNING"
        else:
            state, label = "waiting", "WAITING"
        html.append(f"""
        <div class="pcard {state}">
            <div>
                <span class="pcard-num">{num}</span><span class="pcard-title">{title}</span>
                <div class="pcard-desc">{desc}</div>
            </div>
            <div class="badge {state}">{label}</div>
        </div>
        """)
    return "".join(html)


# ----------------------------------------------------------------------------
# MAIN LAYOUT
# ----------------------------------------------------------------------------
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-label">RESEARCH TOPIC</div>', unsafe_allow_html=True)
    topic = st.text_input(
        "topic", value=st.session_state.topic_input,
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        label_visibility="collapsed",
    )
    run_clicked = st.button("⚡  Run Research Pipeline", disabled=not topic.strip())

    st.markdown('<div class="section-label" style="margin-top:1.5rem;">TRY →</div>', unsafe_allow_html=True)
    chip_cols = st.columns(len(SUGGESTIONS))
    for i, s in enumerate(SUGGESTIONS):
        with chip_cols[i]:
            if st.button(s, key=f"chip_{i}", type="secondary"):
                st.session_state.topic_input = s
                st.rerun()

with right:
    st.markdown('<div class="pipeline-heading">Pipeline</div>', unsafe_allow_html=True)
    pipeline_slot = st.empty()
    pipeline_slot.markdown(render_pipeline_cards(st.session_state.current_step), unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# RUN PIPELINE
# ----------------------------------------------------------------------------
if run_clicked and topic.strip():
    try:
        state = {}

        st.session_state.current_step = 0
        pipeline_slot.markdown(render_pipeline_cards(0), unsafe_allow_html=True)
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = search_result["messages"][-1].content

        st.session_state.current_step = 1
        pipeline_slot.markdown(render_pipeline_cards(1), unsafe_allow_html=True)
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [("user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}"
            )]
        })
        state["scraped_content"] = reader_result["messages"][-1].content

        st.session_state.current_step = 2
        pipeline_slot.markdown(render_pipeline_cards(2), unsafe_allow_html=True)
        research_combined = (
            f"SEARCH RESULT : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})

        st.session_state.current_step = 3
        pipeline_slot.markdown(render_pipeline_cards(3), unsafe_allow_html=True)
        state["feedback"] = critic_chain.invoke({"report": state["report"]})

        st.session_state.current_step = 4
        pipeline_slot.markdown(render_pipeline_cards(4), unsafe_allow_html=True)
        st.session_state.result = state
        st.session_state.topic_input = topic

    except Exception as e:
        st.session_state.current_step = -1
        st.error(f"Pipeline failed: {e}")

# ----------------------------------------------------------------------------
# RESULTS
# ----------------------------------------------------------------------------
if st.session_state.result:
    state = st.session_state.result
    st.markdown('<div class="result-wrap"></div>', unsafe_allow_html=True)
    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["📄 Report", "🧐 Critic Feedback", "🔍 Search Results", "📚 Scraped Content"]
    )
    with tab_report:
        st.markdown(state["report"])
        st.download_button(
            "Download report as .md",
            data=state["report"],
            file_name=f"{st.session_state.topic_input.strip().replace(' ', '_')}_report.md",
            mime="text/markdown",
        )
    with tab_feedback:
        st.markdown(state["feedback"])
    with tab_search:
        st.text(state["search_results"])
    with tab_scraped:
        st.text(state["scraped_content"])