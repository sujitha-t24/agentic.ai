import json
import os
import re
from pathlib import Path
import urllib.error
import urllib.request

import streamlit as st
from ddgs import DDGS

MODEL_DEFAULT = os.getenv("MODEL", "qwen3:4b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/")
SYSTEM_PROMPT = "You are one role in a decision debate. Always respond in English, regardless of the language used in the decision or context. Keep private reasoning private. Output only the requested final response: no planning, self-talk, word-count checks, or narration about tools. Call web_search directly when evidence would help; do not merely say you will search. Never invent studies, statistics, or citations."
TOOLS = [{"type": "function", "function": {"name": "web_search", "description": "Search the web for reliable evidence. Use concise queries.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}}, "required": ["query"]}}}]


def ollama_chat(messages, tools=None, json_mode=False):
    payload = {"model": st.session_state.model, "messages": messages, "stream": False, "think": False, "keep_alive": "10m", "options": {"num_predict": 300}}
    if tools:
        payload["tools"] = tools
    if json_mode:
        payload["format"] = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["Go", "Lean go", "Lean no", "No-go"]}, "confidence": {"type": "integer", "minimum": 0, "maximum": 100}, "summary": {"type": "string"}, "deciding_factor": {"type": "string"}, "key_facts": {"type": "array", "items": {"type": "string"}}}, "required": ["verdict", "confidence", "summary", "deciding_factor", "key_facts"], "additionalProperties": False}
    request = urllib.request.Request(f"{OLLAMA_URL}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode())["message"]
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"Ollama error: {detail}. Check that the model is downloaded (ollama pull {st.session_state.model}).") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError("Could not connect to Ollama. Install and open the Ollama app; setup steps are in README.md.") from exc


def web_search(query, used_queries):
    used_queries.append(query)
    try:
        results = DDGS().text(query, max_results=4)
        return json.dumps([{"title": r.get("title", ""), "url": r.get("href", r.get("url", "")), "snippet": r.get("body", "")} for r in results], ensure_ascii=False)
    except Exception as exc:
        return f"Search unavailable: {exc}. Continue without web evidence and be clear about uncertainty."


def clean_argument(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    points = list(re.finditer(r"(?m)^\s*[1-3][.)]\s+", text))
    if len(points) >= 3:
        start = points[0].start()
        end = len(text)
        tail = text[points[2].end():]
        blank = re.search(r"\n\s*\n", tail)
        if blank:
            end = points[2].end() + blank.start()
        text = text[start:end]
    return text.strip()


def run_agent(name, instruction, decision, context, transcript=""):
    queries = []
    prompt = f"Decision: {decision}\nContext: {context or 'None given'}\n"
    if transcript:
        prompt += f"Full debate transcript so far:\n{transcript}\n\n"
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt + instruction}]
    for _ in range(4):
        reply = ollama_chat(messages, TOOLS)
        calls = reply.get("tool_calls", [])
        messages.append({"role": "assistant", "content": reply.get("content", ""), "tool_calls": calls})
        if not calls:
            answer = reply.get("content", "No response returned.")
            if name == "Judge" and parse_verdict(answer) is None:
                if not queries:
                    query = f"{decision[:140]} evidence fact check"
                    evidence = web_search(query, queries)
                    messages.append({"role": "user", "content": f"App web search results (use as evidence, not as guaranteed truth):\n{evidence}"})
                final = ollama_chat(messages + [{"role": "user", "content": "Now return the final verdict as the required JSON object only."}], json_mode=True)
                answer = final.get("content", answer)
            return (clean_argument(answer) if name in {"Prosecutor", "Defender"} else answer), queries
        for call in calls:
            fn = call.get("function", {})
            if fn.get("name") == "web_search":
                query = fn.get("arguments", {}).get("query", "")
                result = web_search(query, queries) if query else "Search query was empty."
            else:
                result = "Unknown tool."
            messages.append({"role": "tool", "tool_name": fn.get("name", "web_search"), "content": result})
    final = ollama_chat(messages + [{"role": "user", "content": "Give your final answer now. Do not call tools."}], json_mode=name == "Judge")
    answer = final.get("content", "No response returned.")
    return (clean_argument(answer) if name in {"Prosecutor", "Defender"} else answer), queries


def parse_verdict(raw):
    try:
        match = re.search(r"\{.*\}", raw, re.S)
        result = json.loads(match.group(0) if match else raw)
        if result.get("verdict") not in {"Go", "Lean go", "Lean no", "No-go"}:
            return None
        result["confidence"] = max(0, min(100, int(result["confidence"])))
        return result
    except (ValueError, TypeError, KeyError):
        return None


@st.cache_data(show_spinner=False)
def load_averitec():
    path = Path(__file__).parent / "data" / "averitec_dev.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def find_dataset_matches(text, rows, limit=3):
    stop = {"should", "would", "could", "about", "with", "from", "that", "this", "have", "your", "decision", "whether", "into", "when", "what", "does", "doesn", "will", "must", "need", "make", "take", "consider", "considering", "fact", "check", "dataset", "student", "support", "demo", "small", "college", "team", "project", "model", "local", "app"}
    terms = {w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop}
    if not terms:
        return []
    ranked = []
    for row in rows:
        claim = row.get("claim", "")
        words = set(re.findall(r"[a-z]{3,}", claim.lower()))
        overlap = terms & words
        if len(overlap) >= 2:
            score = len(overlap) / (len(terms) * len(words)) ** 0.5
            ranked.append((score, row, sorted(overlap)))
    return [(row, overlap) for _, row, overlap in sorted(ranked, key=lambda item: item[0], reverse=True)[:limit]]


st.set_page_config(page_title="Agent Court", page_icon="⚖️", layout="centered")
averitec_rows = load_averitec()
st.title("⚖️ Agent Court")
st.caption("Three AI agents challenge a decision, look for evidence, and deliver a verdict.")
with st.sidebar:
    st.header("Court settings")
    st.caption("Runs on your computer with Ollama. No Anthropic API key or paid token required.")
    st.session_state.model = st.text_input("Local model", value=st.session_state.get("model", MODEL_DEFAULT), help="Must already be downloaded in Ollama. Default: qwen3:4b")
    rounds = st.slider("Debate rounds", 1, 2, 1)
    st.caption(f"AVeriTeC local reference set: {len(averitec_rows)} labeled examples")
decision = st.text_area("What decision are you considering?", placeholder="Example: Should I switch to a four-day workweek?")
context = st.text_area("Context (optional)", placeholder="Your goals, budget, timeline, or constraints")
if st.button("Convene the court", type="primary", use_container_width=True):
    if not decision.strip():
        st.warning("First, enter a decision for the agents to discuss.")
    elif not st.session_state.model.strip():
        st.warning("Enter a local model name in the sidebar.")
    else:
        roles = [("🔴", "Prosecutor", "In English, give exactly 3 numbered points arguing AGAINST the decision, maximum 150 words total. No introduction or conclusion. Search for evidence where useful; otherwise present concerns as possibilities, not proven facts."), ("🟢", "Defender", "In English, give exactly 3 numbered points arguing FOR the decision and rebutting the Prosecutor, maximum 150 words total. No introduction or conclusion. Search for evidence where useful; do not invent sources."), ("⚖️", "Judge", 'Fact-check the key claims from both sides using web search. Return ONLY valid JSON, with every string value written in English: {"verdict":"Go|Lean go|Lean no|No-go","confidence":65,"summary":"...","deciding_factor":"...","key_facts":["..."]}. Choose an evidence-based confidence from 1 to 100; 65 is only an example, not a default. Do not use 0 as a placeholder. Cite only source URLs returned by web search. No markdown fences.')]
        transcript = []
        try:
            for round_no in range(1, rounds + 1):
                for avatar, name, instruction in roles[:2]:
                    prior = "\n\n".join(transcript) if round_no == 2 else ""
                    with st.chat_message(name.lower(), avatar=avatar):
                        with st.spinner(f"{name} is thinking (round {round_no})..."):
                            answer, queries = run_agent(name, instruction, decision, context, prior)
                        st.markdown(answer)
                        st.caption("Search queries: " + ("; ".join(queries) if queries else "No searches used"))
                    transcript.append(f"Round {round_no} — {name}:\n{answer}")
            with st.chat_message("judge", avatar="⚖️"):
                with st.spinner("Judge is checking claims and reaching a verdict..."):
                    raw, queries = run_agent("Judge", roles[2][2], decision, context, "\n\n".join(transcript))
                st.caption("Search queries: " + ("; ".join(queries) if queries else "No searches used"))
            result = parse_verdict(raw)
            if result:
                st.subheader("Court verdict")
                left, right = st.columns(2)
                left.metric("Verdict", result["verdict"])
                right.metric("Confidence", f"{result['confidence']}%")
                st.progress(result["confidence"])
                st.write(result.get("summary", ""))
                st.info(result.get("deciding_factor", ""))
                for fact in result.get("key_facts", []):
                    st.markdown(f"- {fact}")
            else:
                st.warning("Could not parse the Judge response as JSON. Raw response:")
                st.code(raw)
            matches = find_dataset_matches(decision, averitec_rows)
            st.subheader("AVeriTeC fact-check references")
            st.caption("Similar past claims from the dataset. Keyword matches are examples only; their verdicts do not decide your personal choice.")
            if matches:
                for item, matched_words in matches:
                    title = item.get("claim", "Claim")
                    with st.expander(f"{item.get('label', 'Unlabeled')} — {title[:120]}"):
                        st.caption("Shared keywords: " + ", ".join(matched_words))
                        st.write(item.get("justification", "No justification in this record."))
                        article = item.get("fact_checking_article") or item.get("cached_original_claim_url")
                        if article:
                            st.markdown(f"[Open fact-check source]({article})")
                        for question in item.get("questions", [])[:2]:
                            st.markdown(f"**Question:** {question.get('question', '')}")
                            for answer in question.get("answers", [])[:1]:
                                st.write(answer.get("answer", ""))
                                if answer.get("source_url"):
                                    st.markdown(f"[Evidence source]({answer['source_url']})")
            else:
                st.info("No close keyword matches found. The dataset references are optional and don't affect the live verdict.")
        except Exception as exc:
            st.error(f"The app could not complete this run: {exc}")
            st.info("Check that Ollama is running, the model is downloaded, and internet search is available. See README.md for setup steps.")
