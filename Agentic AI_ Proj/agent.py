# agent.py
# Agentic AI Course Assistant — Full LangGraph Agent
# Author: [Soumik Pal] | Roll No: [2328206] | Batch: Agentic AI 2026
#
# This is the main agent file. It wires together the knowledge base,
# the LLM, and all the graph nodes into one compiled LangGraph app.
# Run this file directly for a quick smoke test:
#   python agent.py
from dotenv import load_dotenv
load_dotenv()
import os
import re
import uuid
from datetime import datetime
from typing import TypedDict, List

# LangGraph gives us the graph structure and memory persistence
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Groq runs Llama-3 fast and free — perfect for the course tier
from langchain_groq import ChatGroq

# ChromaDB is the vector database; SentenceTransformer creates the embeddings
import chromadb
from sentence_transformers import SentenceTransformer

# Our 12-document knowledge base covering every major course topic
from knowledge_base import documents


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Fail fast with a clear message if the API key is missing.
# Much better than getting a cryptic auth error 10 nodes deep in the graph.
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY environment variable is not set.\n"
        "Fix it with:  export GROQ_API_KEY='your_key_here'"
    )

# How many times the eval node is allowed to send the answer back for a retry
# before we accept whatever answer we have and move on.
MAX_EVAL_RETRIES = 2

# Keyword → topic name map used by memory_node to detect what the student is
# asking about. The first matching keyword wins, so more specific terms
# (e.g. "memorysaver") are listed before generic ones (e.g. "memory").
TOPIC_KEYWORDS = {
    "agentic ai":      "What is Agentic AI",
    "langgraph":       "LangGraph StateGraph Nodes and Edges",
    "stategraph":      "LangGraph StateGraph Nodes and Edges",
    "typeddict":       "State Design with TypedDict",
    "state design":    "State Design with TypedDict",
    "memorysaver":     "Memory with MemorySaver and thread_id",
    "thread_id":       "Memory with MemorySaver and thread_id",
    "memory":          "Memory with MemorySaver and thread_id",
    "chromadb":        "ChromaDB RAG Setup and Retrieval",
    "embedding":       "ChromaDB RAG Setup and Retrieval",
    "rag":             "ChromaDB RAG Setup and Retrieval",
    "retrieval":       "ChromaDB RAG Setup and Retrieval",
    "router":          "Router Node and Routing Logic",
    "routing":         "Router Node and Routing Logic",
    "faithfulness":    "Eval Node and Self-Reflection",
    "eval":            "Eval Node and Self-Reflection",
    "self-reflection": "Eval Node and Self-Reflection",
    "datetime":        "Tool Use — DateTime and Calculator",
    "calculator":      "Tool Use — DateTime and Calculator",
    "tool":            "Tool Use — DateTime and Calculator",
    "cache_resource":  "Streamlit Deployment and Session State",
    "session state":   "Streamlit Deployment and Session State",
    "streamlit":       "Streamlit Deployment and Session State",
    "ragas":           "RAGAS Evaluation Framework",
    "evaluation":      "RAGAS Evaluation Framework",
    "red-team":        "Red-Teaming and Adversarial Testing",
    "red team":        "Red-Teaming and Adversarial Testing",
    "adversarial":     "Red-Teaming and Adversarial Testing",
    "submission":      "Project Structure and Submission Checklist",
    "checklist":       "Project Structure and Submission Checklist",
    "project":         "Project Structure and Submission Checklist",
    "node":            "LangGraph StateGraph Nodes and Edges",
    "edge":            "LangGraph StateGraph Nodes and Edges",
    "state":           "State Design with TypedDict",
}


# ══════════════════════════════════════════════════════════════════════════════
# STATE DESIGN  —  always defined first, before any node function
#
# Every field that any node reads or writes must be declared here.
# Missing a field causes a KeyError at runtime — there is no partial update.
# Think of this as the contract the entire graph depends on.
# ══════════════════════════════════════════════════════════════════════════════

class CourseAssistantState(TypedDict):
    question:     str         # what the student just asked
    messages:     List[dict]  # full conversation history (sliding window applied)
    route:        str         # router's decision: "retrieve" | "tool" | "memory_only"
    retrieved:    str         # formatted context string from ChromaDB retrieval
    sources:      List[str]   # topic names of the retrieved chunks
    tool_result:  str         # string output from the tool node
    answer:       str         # final generated answer from the LLM
    faithfulness: float       # eval score ranging from 0.0 (hallucinated) to 1.0 (grounded)
    eval_retries: int         # how many answer retries have happened so far
    user_name:    str         # student's name, extracted from "my name is X"
    topic_asked:  str         # which course topic the question maps to


# ══════════════════════════════════════════════════════════════════════════════
# INITIALISATION — build the KB and the LLM once, share across all nodes
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_base():
    """
    Loads the embedding model, builds the ChromaDB collection, and runs
    a quick retrieval sanity check before the graph is assembled.

    We test retrieval here because a broken knowledge base cannot be fixed
    by improving the LLM prompt — it must be caught before the graph runs.
    """
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Building ChromaDB in-memory collection...")
    client = chromadb.Client()

    # Delete and recreate so this function is safe to call more than once
    try:
        client.delete_collection("course_kb")
    except Exception:
        pass

    collection = client.create_collection("course_kb")

    texts = [doc["text"]          for doc in documents]
    ids   = [doc["id"]            for doc in documents]
    metas = [{"topic": doc["topic"]} for doc in documents]

    embeddings = embedder.encode(texts).tolist()
    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metas,
    )
    print(f"  ✓ {len(documents)} documents indexed successfully")

    # Sanity check — make sure we can retrieve something relevant
    test_query = "how does LangGraph memory work"
    test_emb   = embedder.encode([test_query]).tolist()
    test_res   = collection.query(query_embeddings=test_emb, n_results=2)
    topics     = [m["topic"] for m in test_res["metadatas"][0]]
    print(f"  ✓ Retrieval sanity check: '{test_query}' → {topics}\n")

    return embedder, collection


def build_llm():
    """Creates the Groq LLM client. Temperature 0.2 keeps answers factual."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
    )


# ══════════════════════════════════════════════════════════════════════════════
# NODE FUNCTIONS
#
# Each node is a plain Python function that receives the current state dict
# and returns a dict of fields it wants to update. LangGraph merges the
# return dict into the state — it's a partial update, not a full replacement.
#
# Nodes that depend on external resources (llm, embedder, collection) are
# written as closures — make_X_node() returns the actual node function with
# its dependencies baked in. No globals needed.
# ══════════════════════════════════════════════════════════════════════════════

def make_memory_node():

    def memory_node(state: CourseAssistantState) -> dict:
        """
        First node every turn. Responsible for three things:
          1. Append the user's question to the conversation history.
          2. Apply a 6-message sliding window to stay within Groq's token limit.
          3. Extract the student's name if they said "my name is X".
          4. Detect which course topic the question is about (topic_asked).

        Also resets all per-turn fields so data from the previous turn
        doesn't accidentally leak into this one.
        """
        question  = state.get("question", "")
        messages  = list(state.get("messages", []))
        user_name = state.get("user_name", "")

        # Add the current question to the history first
        messages.append({"role": "user", "content": question})

        # Keep only the last 6 messages — Groq free tier has token limits
        messages = messages[-6:]

        # Look for "my name is X" anywhere in the message
        name_match = re.search(r"my name is ([A-Za-z]+)", question, re.IGNORECASE)
        if name_match:
            user_name = name_match.group(1).strip()

        # Scan for known topic keywords so downstream nodes know what's being asked
        q_lower     = question.lower()
        topic_asked = ""
        for keyword, topic in TOPIC_KEYWORDS.items():
            if keyword in q_lower:
                topic_asked = topic
                break  # first (most specific) match wins

        return {
            "messages":    messages,
            "user_name":   user_name,
            "topic_asked": topic_asked,
            # Clear all per-turn working fields before the new turn begins
            "retrieved":    "",
            "sources":      [],
            "tool_result":  "",
            "answer":       "",
            "faithfulness": 0.0,
            "eval_retries": 0,
        }

    return memory_node


def make_router_node(llm):

    def router_node(state: CourseAssistantState) -> dict:
        """
        Reads the student's question and decides which path to take.
        The LLM replies with exactly one word — retrieve, tool, or memory_only.
        If it returns anything else, we default to retrieve (safest fallback).

        - retrieve    → question needs content from the knowledge base
        - tool        → question needs the current date/time or arithmetic
        - memory_only → question can be answered from conversation history alone
        """
        question = state.get("question", "")

        prompt = f"""You are a routing assistant for an Agentic AI course chatbot.
Classify the user's question into exactly ONE of these three routes:

  retrieve    — the question asks about course topics (LangGraph, RAG, memory, tools,
                RAGAS, Streamlit, nodes, state design, evaluation, project structure, etc.)

  tool        — the question needs the current date, time, or simple arithmetic
                (examples: "what is today's date?", "what is 25 multiplied by 4?")

  memory_only — the question is purely conversational and can be answered from the
                chat history alone (examples: "what did I just ask?", "what's my name?")

When in doubt, choose retrieve — it is always safer to retrieve and find nothing
than to skip retrieval for a question that actually needs it.

Reply with ONLY ONE WORD — no punctuation, no explanation.

User question: {question}
Route:"""

        response = llm.invoke(prompt)
        raw      = response.content.strip().lower().split()[0]

        # Sanitise the output — if LLM returns something unexpected, default to retrieve
        route = raw if raw in ("retrieve", "tool", "memory_only") else "retrieve"
        print(f"  [router_node] route='{route}'")
        return {"route": route}

    return router_node


def make_retrieval_node(embedder, collection):

    def retrieval_node(state: CourseAssistantState) -> dict:
        """
        Embeds the question using the same model that indexed the KB,
        then queries ChromaDB for the top 3 most relevant chunks.

        The chunks are formatted with [Topic] labels so the answer node
        knows exactly where each piece of information came from.
        """
        question = state.get("question", "")

        q_embedding = embedder.encode([question]).tolist()
        results     = collection.query(query_embeddings=q_embedding, n_results=3)

        chunks  = results["documents"][0]
        metas   = results["metadatas"][0]
        sources = [m["topic"] for m in metas]

        # Format with topic headers so the LLM can cite its sources clearly
        context_parts = [f"[{topic}]\n{chunk}" for topic, chunk in zip(sources, chunks)]
        retrieved     = "\n\n---\n\n".join(context_parts)

        print(f"  [retrieval_node] retrieved from: {sources}")
        return {"retrieved": retrieved, "sources": sources}

    return retrieval_node


def skip_retrieval_node(state: CourseAssistantState) -> dict:
    """
    Used on the memory_only route — no KB lookup needed.
    Returns empty fields so answer_node falls back to conversation history.
    """
    return {"retrieved": "", "sources": []}


def make_tool_node():

    def tool_node(state: CourseAssistantState) -> dict:
        """
        Handles two tools: datetime and calculator.

        The absolute rule for tools: they must NEVER raise an exception.
        If a tool crashes, it crashes the entire graph run. Every code path
        here ends in a return statement with a user-readable string.

        Calculator guard: we require at least one digit in the question before
        attempting arithmetic. This prevents false matches on natural-language
        "what is X" questions that the router mistakenly sent here.
        """
        question = state.get("question", "").lower()
        result   = ""

        try:
            # ── DateTime tool ──────────────────────────────────────────────
            if any(kw in question for kw in ["date", "time", "today", "day", "when"]):
                now    = datetime.now()
                result = (
                    f"Current date and time: {now.strftime('%A, %d %B %Y')}, "
                    f"{now.strftime('%I:%M %p')}. "
                    f"Your capstone project deadline is April 21, 2026 at 11:59 PM."
                )

            # ── Calculator tool ────────────────────────────────────────────
            # Only attempt arithmetic if there is at least one digit present
            elif re.search(r"\d", question) and any(
                kw in question for kw in ["calculate", "compute", "+", "-", "*", "/", "=", "percent", "%"]
            ):
                # Strip everything except digits, operators, and parentheses
                expr = re.sub(r"[^0-9+\-*/().\s]", "", question.replace("%", "/100"))
                expr = expr.strip()

                if expr:
                    # eval is sandboxed — no builtins, so no import/exec attacks
                    value  = eval(expr, {"__builtins__": {}})  # noqa: S307
                    result = f"The result of {expr} is {round(value, 4)}."
                else:
                    result = "I couldn't find a clear arithmetic expression. Please rephrase."

            else:
                result = (
                    "I couldn't identify which tool to use for this question. "
                    "Try asking about a course topic instead."
                )

        except ZeroDivisionError:
            result = "That expression involves a division by zero, which is undefined."
        except Exception as e:
            result = f"The tool ran into an error ({str(e)}). Please try rephrasing."

        print(f"  [tool_node] result='{result[:80]}...'")
        return {"tool_result": result}

    return tool_node


def make_answer_node(llm):

    def answer_node(state: CourseAssistantState) -> dict:
        """
        Generates the final answer. Uses whichever context is available:
          - retrieved: content from ChromaDB (retrieval route)
          - tool_result: output from the tool node (tool route)
          - conversation history only (memory_only route)

        On a retry (eval_retries >= 1), the prompt includes an escalation
        instruction telling the LLM it was previously flagged and must be
        more conservative. eval_retries is incremented here so that
        eval_decision can read the updated count on the next pass.
        """
        question     = state.get("question", "")
        retrieved    = state.get("retrieved", "")
        tool_result  = state.get("tool_result", "")
        messages     = state.get("messages", [])
        user_name    = state.get("user_name", "")
        eval_retries = state.get("eval_retries", 0)

        # Build a readable conversation history string (excluding the current question)
        history_lines = []
        for msg in messages[:-1]:
            label = "Student" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{label}: {msg['content']}")
        history_str = "\n".join(history_lines) if history_lines else "No prior conversation."

        # Personalise the response if we know the student's name
        name_line = (
            f"The student's name is {user_name}. Address them by name in your response."
            if user_name else ""
        )

        # Escalation warning shown on retries so the LLM knows to be more careful
        retry_instruction = ""
        if eval_retries >= 1:
            retry_instruction = (
                "\n⚠️  RETRY ALERT: Your previous answer scored below the faithfulness "
                "threshold. This means it likely contained information not present in "
                "the context. Be extra conservative this time — only state what is "
                "explicitly written in the context below. If the answer isn't there, "
                "say so clearly rather than guessing."
            )

        # Pick the right context block based on which route we came from
        if retrieved:
            context_section = f"COURSE KNOWLEDGE BASE CONTEXT:\n{retrieved}"
        elif tool_result:
            context_section = f"TOOL RESULT:\n{tool_result}"
        else:
            context_section = (
                "No external context is available. "
                "Answer using the conversation history only."
            )

        system_prompt = f"""You are a friendly and helpful course assistant for the Agentic AI \
course taught by Dr. Kanthi Kiran Sirra.
{name_line}

GOLDEN RULE: Answer ONLY using the course context provided below. Do NOT draw on outside knowledge.
If the answer is not in the provided context, say exactly this:
"I don't have information on that in my course materials. Please contact Dr. Kanthi Kiran Sirra \
or check your course notebooks for more detail."

Additional rules:
- Never give medical, legal, or financial advice.
- If the student seems stressed or distressed, respond with empathy and encouragement first.
- Keep answers clear, concise, and student-friendly.
- Do not make up course content, document names, or instructor details.
{retry_instruction}

CONVERSATION HISTORY:
{history_str}

{context_section}

Student's question: {question}

Your answer:"""

        response = llm.invoke(system_prompt)

        # Increment eval_retries here so eval_decision sees the updated count.
        # This is the correct place — routing functions cannot write to state.
        return {
            "answer":       response.content.strip(),
            "eval_retries": eval_retries + 1,
        }

    return answer_node


def make_eval_node(llm):

    def eval_node(state: CourseAssistantState) -> dict:
        """
        Scores the answer's faithfulness against the retrieved context.
        Returns a float from 0.0 (hallucinated) to 1.0 (fully grounded).

        Skipped entirely when there is no retrieved context — this happens
        on tool and memory_only routes where faithfulness scoring makes no
        sense (there is no context to be faithful to).

        This node does NOT increment eval_retries. That happens inside
        answer_node so the counter accurately reflects completed retries.
        """
        retrieved = state.get("retrieved", "")
        answer    = state.get("answer", "")
        question  = state.get("question", "")

        # Nothing to evaluate against — treat as fully faithful and move on
        if not retrieved.strip():
            print("  [eval_node] no context to evaluate — skipping (score=1.0)")
            return {"faithfulness": 1.0}

        eval_prompt = f"""You are an impartial evaluator assessing whether an AI answer \
is faithfully grounded in the provided context.

Context (the only source the answer should use):
{retrieved}

Question that was asked:
{question}

Answer to evaluate:
{answer}

Scoring guide:
  1.0 — every single claim in the answer is directly supported by the context
  0.8 — mostly grounded with very minor extrapolation
  0.6 — roughly half the answer uses information from outside the context
  0.4 — most claims are from outside the context or are vague paraphrases
  0.0 — the answer ignores the context entirely or fabricates information

Reply with ONLY a single decimal number between 0.0 and 1.0.
No words, no explanation, just the number.
Score:"""

        try:
            response  = llm.invoke(eval_prompt)
            score_str = response.content.strip().split()[0]
            score     = float(score_str)
            score     = max(0.0, min(1.0, score))  # clamp to valid range
        except Exception:
            # If the LLM returns something we can't parse, use a neutral score
            # that doesn't trigger a retry but doesn't claim perfection either
            score = 0.75
            print("  [eval_node] could not parse score — using fallback 0.75")

        print(f"  [eval_node] faithfulness={score:.2f}")
        return {"faithfulness": score}

    return eval_node


def save_node(state: CourseAssistantState) -> dict:
    """
    Last node before END. Appends the assistant's answer to the message
    history and re-applies the sliding window so the list never exceeds
    6 messages (without this, the window would grow to 7 after the
    assistant turn is added).
    """
    messages = list(state.get("messages", []))
    answer   = state.get("answer", "")

    messages.append({"role": "assistant", "content": answer})
    messages = messages[-6:]  # keep the window consistent after assistant turn

    return {"messages": messages}


# ══════════════════════════════════════════════════════════════════════════════
# ROUTING FUNCTIONS
#
# These are called by LangGraph's conditional edges to decide the next node.
# They are READ-ONLY — any mutation to state here is silently dropped by
# LangGraph. State updates must happen inside node functions, not here.
# ══════════════════════════════════════════════════════════════════════════════

def route_decision(state: CourseAssistantState) -> str:
    """
    Called after router_node. Maps state.route to the next node name.
    Returns one of: "retrieve", "skip", "tool"
    """
    route = state.get("route", "retrieve")
    mapping = {"tool": "tool", "memory_only": "skip"}
    return mapping.get(route, "retrieve")


def eval_decision(state: CourseAssistantState) -> str:
    """
    Called after eval_node. Decides whether to retry the answer or save it.

    The retry condition: faithfulness < 0.7 AND we haven't hit the max yet.
    eval_retries was already incremented by answer_node, so by the time we
    read it here it reflects the number of answers that have been generated
    (including the one we just evaluated).

    Returns "answer" to retry, or "save" to accept and finish.
    """
    faithfulness = state.get("faithfulness", 1.0)
    eval_retries = state.get("eval_retries", 0)

    if faithfulness < 0.7 and eval_retries <= MAX_EVAL_RETRIES:
        print(f"  [eval_decision] RETRY — faithfulness={faithfulness:.2f}, retries so far={eval_retries}")
        return "answer"
    else:
        print(f"  [eval_decision] PASS  — faithfulness={faithfulness:.2f}, retries so far={eval_retries}")
        return "save"


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_graph(llm, embedder, collection):
    """
    Wires all the nodes together into a compiled LangGraph app.

    Node execution order:
      memory → router → [retrieve | skip | tool] → answer → eval → save → END

    The two conditional branches are:
      - After router: route_decision picks retrieve / skip / tool
      - After eval:   eval_decision picks answer (retry) or save (done)
    """

    # Build node closures with their dependencies injected
    memory_node    = make_memory_node()
    router_node    = make_router_node(llm)
    retrieval_node = make_retrieval_node(embedder, collection)
    tool_node      = make_tool_node()
    answer_node    = make_answer_node(llm)
    eval_node      = make_eval_node(llm)

    graph = StateGraph(CourseAssistantState)

    # Register every node
    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    # The graph always starts at memory
    graph.set_entry_point("memory")

    # Fixed edges — these never change based on state
    graph.add_edge("memory",   "router")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")
    graph.add_edge("save",     END)

    # Conditional edges — these use routing functions to decide the next node
    graph.add_conditional_edges(
        "router",
        route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"},
    )
    graph.add_conditional_edges(
        "eval",
        eval_decision,
        {"answer": "answer", "save": "save"},
    )

    # compile() validates the graph structure and attaches the memory checkpointer
    app = graph.compile(checkpointer=MemorySaver())
    print("✓ Graph compiled successfully\n")
    return app


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC HELPER — used by capstone_streamlit.py and tests
# ══════════════════════════════════════════════════════════════════════════════

def ask(app, question: str, thread_id: str = "default") -> dict:
    """
    Thin wrapper around app.invoke() that handles the config boilerplate.

    Args:
        app:       the compiled LangGraph app from build_graph()
        question:  the student's question as a plain string
        thread_id: conversation session ID — same ID = shared memory

    Returns a dict with:
        answer       — the agent's response text
        route        — which route the router chose
        faithfulness — the eval score (0.0 to 1.0)
        sources      — list of KB topic names used in retrieval
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"question": question}, config=config)
    return {
        "answer":       result.get("answer", ""),
        "route":        result.get("route", ""),
        "faithfulness": result.get("faithfulness", 0.0),
        "sources":      result.get("sources", []),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SMOKE TEST — run with: python agent.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  Agentic AI Course Assistant — Smoke Test")
    print("=" * 60 + "\n")

    embedder, collection = build_knowledge_base()
    llm = build_llm()
    app = build_graph(llm, embedder, collection)

    # Use a fresh UUID each run so previous smoke-test memory doesn't bleed in
    tid = str(uuid.uuid4())
    print(f"Thread ID: {tid}\n")

    test_questions = [
        # Retrieval tests
        ("What is Agentic AI and how is it different from a regular LLM?", "retrieve"),
        ("My name is Arjun. Can you explain what MemorySaver does?",        "retrieve"),
        ("What does the eval node check for?",                               "retrieve"),
        ("How do I set up ChromaDB for the knowledge base?",                 "retrieve"),
        # Tool test
        ("What is today's date?",                                            "tool"),
        # Memory test — must recall name from turn 2 above
        ("What is my name?",                                                 "memory_only"),
        # Red-team test — must refuse cleanly
        ("What is the capital of Japan?",                                    "retrieve"),
    ]

    print("=" * 60)
    print("  Running tests")
    print("=" * 60 + "\n")

    passed = 0
    for i, (question, expected_route) in enumerate(test_questions, 1):
        print(f"[Q{i}] {question}")
        result = ask(app, question, thread_id=tid)

        route_ok = result["route"] == expected_route
        status   = "PASS" if route_ok else "WARN"
        if route_ok:
            passed += 1

        print(f"  Route:       {result['route']} (expected: {expected_route}) → {status}")
        print(f"  Faithfulness:{result['faithfulness']:.2f}")
        print(f"  Sources:     {result['sources']}")
        print(f"  Answer:      {result['answer'][:180]}...")
        print()

    print("=" * 60)
    print(f"  Results: {passed}/{len(test_questions)} route checks passed")
    print("=" * 60)