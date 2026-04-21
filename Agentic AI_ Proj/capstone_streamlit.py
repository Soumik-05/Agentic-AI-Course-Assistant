# capstone_streamlit.py
# Agentic AI Course Assistant — Streamlit UI
# Author: [Soumik Pal] | Roll No: [2328206] | Batch: Agentic AI 2026
#
# Launch: streamlit run capstone_streamlit.py

import uuid
import streamlit as st

# ── Import agent components ────────────────────────────────────────────────
from agent import build_knowledge_base, build_llm, build_graph, ask


# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Agentic AI Course Assistant",
    page_icon="🤖",
    layout="wide",
)


# ══════════════════════════════════════════════════════════════════════════════
# CACHED RESOURCES — initialised ONCE per server session (not on every rerun)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource
def load_agent():
    """
    Heavy initialisation lives here.
    @st.cache_resource ensures this runs only once — not on every user interaction.
    Returns the compiled LangGraph app so it can be reused across reruns.
    """
    embedder, collection = build_knowledge_base()
    llm  = build_llm()
    app  = build_graph(llm, embedder, collection)
    return app


app = load_agent()


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE — persists across reruns within one browser session
# ══════════════════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/robot-2.png", width=80)
    st.title("Course Assistant")
    st.caption("Agentic AI 2026 · Dr. Kanthi Kiran Sirra")

    st.markdown("---")
    st.markdown("**📚 Topics I can help with:**")
    topics = [
        "What is Agentic AI",
        "LangGraph — StateGraph & nodes",
        "State design with TypedDict",
        "Memory — MemorySaver & thread_id",
        "ChromaDB RAG setup",
        "Router node & routing logic",
        "Eval node & self-reflection",
        "Tool use (datetime, calculator)",
        "Streamlit deployment",
        "RAGAS evaluation",
        "Red-teaming & adversarial tests",
        "Project structure & checklist",
    ]
    for t in topics:
        st.markdown(f"• {t}")

    st.markdown("---")

    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.markdown("---")
    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")
    st.caption("⚠️ I only answer questions about this course. For anything else, contact your instructor.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

st.title("🤖 Agentic AI Course Assistant")
st.caption("Ask me anything about the Agentic AI course — LangGraph, RAG, memory, tools, RAGAS, and more.")

# ── Render conversation history ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Show metadata for assistant messages if available
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]
            cols = st.columns(3)
            cols[0].caption(f"Route: `{meta.get('route', '—')}`")
            cols[1].caption(f"Faithfulness: `{meta.get('faithfulness', 0):.2f}`")
            sources = meta.get("sources", [])
            cols[2].caption(f"Sources: `{', '.join(sources[:2]) if sources else '—'}`")

# ── Welcome message on first load ─────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hello! I'm your **Agentic AI Course Assistant**.\n\n"
            "I can answer questions about everything covered in this course — "
            "LangGraph, ChromaDB, memory, tools, RAGAS evaluation, red-teaming, "
            "and your capstone project structure.\n\n"
            "Feel free to tell me your name so I can address you personally. "
            "What would you like to know?"
        )

# ── Chat input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a course question..."):

    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Call the agent
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(app, prompt, thread_id=st.session_state.thread_id)

        answer      = result["answer"]
        route       = result["route"]
        faithfulness = result["faithfulness"]
        sources     = result["sources"]

        st.markdown(answer)

        # Inline metadata row
        cols = st.columns(3)
        cols[0].caption(f"Route: `{route}`")
        cols[1].caption(f"Faithfulness: `{faithfulness:.2f}`")
        cols[2].caption(f"Sources: `{', '.join(sources[:2]) if sources else '—'}`")

    # Save to session state with metadata
    st.session_state.messages.append({
        "role":    "assistant",
        "content": answer,
        "meta": {
            "route":       route,
            "faithfulness": faithfulness,
            "sources":     sources,
        }
    })
