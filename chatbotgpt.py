# app.py
import os
import json
import time
import streamlit as st
from typing import List, Dict
try:
    import openai
except Exception:
    st.error("openai package not installed. See requirements.txt and install.")
    raise

# ---------- Configuration ----------
APP_TITLE = "Acme AI Chat — Client Demo"
BRAND = {
    "company_name": "Acme AI Solutions",
    "tagline": "Conversational AI for your business",
    # small logo url if you host one; optional
    "logo_url": ""
}
DEFAULT_MODEL = "gpt-4o-mini"  # change according to your OpenAI access
MAX_HISTORY = 40

# ---------- Helpers ----------
def get_api_key():
    """
    Secure: read from environment variable OPENAI_API_KEY, else Streamlit secrets.
    Do NOT hardcode keys.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        # Streamlit secrets
        try:
            key = st.secrets["OPENAI_API_KEY"]
        except Exception:
            key = None
    return key

def init_openai():
    key = get_api_key()
    if not key:
        st.error(
            "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
            "or add it to .streamlit/secrets.toml. See instructions in the README section below."
        )
        st.stop()
    openai.api_key = key

def default_system_prompt():
    return (
        "You are an expert assistant for a company called Acme AI Solutions. "
        "Be professional, concise, and provide clear next steps when appropriate."
    )

def append_message(history: List[Dict], role: str, content: str):
    history.append({"role": role, "content": content})
    # keep size bounded
    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

def chat_completion_call(model: str, messages: List[Dict], temperature: float, max_tokens: int=1024):
    """
    Minimal wrapper for OpenAI ChatCompletion (chat completion API).
    This code uses the openai.ChatCompletion.create endpoint. Adjust depending on the
    model you use (gpt-4, gpt-4o, gpt-3.5-turbo, etc.).
    """
    # Example with ChatCompletion API
    resp = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
    )
    return resp

# ---------- Streamlit UI ----------
st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# Top bar / branding
col1, col2 = st.columns([0.14, 0.86])
with col1:
    if BRAND.get("logo_url"):
        st.image(BRAND["logo_url"], width=64)
with col2:
    st.markdown(f"### {BRAND['company_name']}  ·  {BRAND['tagline']}")

# Sidebar: settings and controls
st.sidebar.header("Chat Controls")
model = st.sidebar.selectbox("Model", options=[DEFAULT_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.2, 0.2, 0.01)
max_tokens = st.sidebar.slider("Max response tokens", 256, 4096, 1024, step=256)
system_prompt = st.sidebar.text_area("System prompt (assistant instructions)", value=default_system_prompt(), height=120)
persona = st.sidebar.selectbox("Persona preset", ["Professional assistant", "Customer support rep", "Technical expert"])
if persona == "Customer support rep":
    persona_prompt = "You are a friendly, patient customer support representative answering with empathy and clarity."
elif persona == "Technical expert":
    persona_prompt = "You are a highly technical engineer; include precise steps, code snippets if relevant."
else:
    persona_prompt = "You are professional and concise."

# file upload for extra context
st.sidebar.header("Context / Files")
uploaded_files = st.sidebar.file_uploader("Upload documents to include as context (.txt, .pdf recommended)", accept_multiple_files=True)
enable_summarize_uploads = st.sidebar.checkbox("Auto-summarize uploaded text files (light)", True)

# conversation controls
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of messages dicts: {"role":..., "content":...}
if "display_history" not in st.session_state:
    st.session_state.display_history = []  # list of (role, content)

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = system_prompt

# show any uploaded file names
if uploaded_files:
    st.sidebar.markdown("**Uploaded files:**")
    for f in uploaded_files:
        st.sidebar.markdown(f"- {f.name} ({f.type or 'unknown'})")

# Main layout: chat area and controls
left, right = st.columns([0.7, 0.3])

with left:
    st.header("Chat")
    # display conversation
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            st.markdown(f"**System:** {content}")
        elif role == "assistant":
            st.chat_message("assistant").write(content)
        else:
            st.chat_message("user").write(content)

    # user input
    user_input = st.chat_input("Type your message here...")
    if user_input:
        # initialize openai
        init_openai()

        # incorporate persona + system prompt
        st.session_state.system_prompt = system_prompt + "\n" + persona_prompt

        # prepare messages: system first, then history, then user
        messages = []
        messages.append({"role": "system", "content": st.session_state.system_prompt})
        # include any uploaded text files as assistant-visible context
        if uploaded_files:
            for f in uploaded_files:
                try:
                    # attempt read text
                    raw = f.getvalue()
                    if isinstance(raw, bytes):
                        try:
                            raw_text = raw.decode("utf-8")
                        except Exception:
                            raw_text = ""
                    else:
                        raw_text = str(raw)
                    if raw_text:
                        if enable_summarize_uploads:
                            # short summary: first 1000 chars as lightweight summary to avoid huge payload
                            snippet = raw_text[:1500]
                            messages.append({"role": "system", "content": f"Context from uploaded file '{f.name}':\n{snippet}"})
                        else:
                            messages.append({"role": "system", "content": f"Context from uploaded file '{f.name}':\n{raw_text[:4000]}"})
                except Exception as e:
                    # skip file on error
                    messages.append({"role": "system", "content": f"Could not read uploaded file '{f.name}' (skipped)."})
        # append prior user/assistant messages
        for m in st.session_state.chat_history:
            messages.append({"role": m["role"], "content": m["content"]})
        # append this user message
        messages.append({"role": "user", "content": user_input})
        # append to UI history immediately
        append_message(st.session_state.chat_history, "user", user_input)

        # show typing indicator
        with st.spinner("Assistant is typing..."):
            try:
                resp = chat_completion_call(model=model, messages=messages, temperature=float(temperature), max_tokens=max_tokens)
                # parse response text
                # ChatCompletion response shape: choices[0].message.content
                assistant_text = ""
                if resp and "choices" in resp and len(resp["choices"]) > 0:
                    assistant_text = resp["choices"][0]["message"]["content"]
                else:
                    assistant_text = "No response from the model."

                append_message(st.session_state.chat_history, "assistant", assistant_text)
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error calling OpenAI API: {e}")

with right:
    st.header("Controls & Export")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    st.download_button("Export conversation (JSON)", data=json.dumps(st.session_state.chat_history, indent=2), file_name="conversation.json", mime="application/json")

    st.markdown("---")
    st.markdown("**Quick prompts (company)**")
    if st.button("Introduce our product"):
        quick = (
            "Write a short introduction (2-3 paragraphs) presenting Acme AI Solutions' chatbot product for a corporate client. "
            "Include value props: faster support, better analytics, and secure deployment. Provide a 3-bullet 'next steps' section."
        )
        append_message(st.session_state.chat_history, "user", quick)
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("**Model debug / raw**")
    if st.checkbox("Show raw session messages"):
        st.json(st.session_state.chat_history)

# Footer / small guidance
st.markdown("---")
st.caption("Built for demo / client handoff. For production, deploy behind a secure secret manager and enable rate limiting, logging, and monitoring.")

# end of file
