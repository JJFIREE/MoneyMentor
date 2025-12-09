# user_pages/chatbot.py
# Chat UI for MoneyMentor - polished + fixed Firestore usage + better visuals
import os
import time
import re
import json
import uuid
import logging
from datetime import datetime

import plotly.graph_objs as go
import streamlit as st
from plotly.io import to_json, from_json

# Import chatbot backend (your ChatBot folder)
from ChatBot.chatbot import FinancialChatBot

# Firebase/Auth helpers (you already had these)
import auth_functions
from auth_functions import initialize_firebase_once, db, sign_out, firestore  # ensure firestore is available

# Initialize firebase once (if your helper does it idempotently)
initialize_firebase_once()

logger = logging.getLogger(__name__)

# ---------- Helper functions for Firestore (fixed & consistent) ----------
def create_new_chat(user_email: str) -> str:
    """
    Create a new chat document in Firestore under collection 'Chats'.
    Chat ID is userId + timestamp to keep things unique and traceable.
    (If you prefer UUID, switch to uuid.uuid4()).
    """
    user_id = st.session_state.get("user_id", "anonymous")
    # deterministic-ish id: user + epoch seconds
    chat_id = f"{user_id}_{int(time.time())}"
    try:
        chat_ref = db.collection("Chats").document(chat_id)
        chat_ref.set({
            "metadata": {
                "userId": user_id,
                "userEmail": user_email,
                "createdAt": firestore.SERVER_TIMESTAMP,
                "lastUpdated": firestore.SERVER_TIMESTAMP,
            }
        })
    except Exception as e:
        logger.exception("Failed to create chat doc: %s", e)
    return chat_id


def save_message(chat_id: str, message: str, sender: str) -> None:
    """
    Save one message to the chat subcollection.
    Collection used: 'Chats' (document chat_id) -> subcollection 'messages'.
    Timestamp field name: 'sentAt' (consistent everywhere).
    """
    try:
        chat_ref = db.collection("Chats").document(chat_id)
        chat_ref.collection("messages").add({
            "content": message,
            "role": sender,
            "sentAt": firestore.SERVER_TIMESTAMP,
        })
        # updating metadata; wrap in try in case permissions or missing doc
        try:
            chat_ref.update({"metadata.lastUpdated": firestore.SERVER_TIMESTAMP})
        except Exception:
            # If update fails, ignore (non-fatal)
            pass
    except Exception as e:
        logger.exception("Failed saving message: %s", e)


def get_chat_history(chat_id: str):
    """
    Return ordered list of messages for a chat.
    Order by 'sentAt' which we set in save_message.
    """
    try:
        messages_ref = db.collection("Chats").document(chat_id).collection("messages")
        messages = messages_ref.order_by("sentAt").stream()
        return [m.to_dict() for m in messages]
    except Exception as e:
        logger.exception("Failed fetching chat history: %s", e)
        return []


# ---------- UI helper: clean assistant text for display ----------
def clean_assistant_text(raw_text: str) -> str:
    """
    - Replace literal escape sequences (like \\n) with real newlines
    - Collapse >2 blank lines into 2
    - Convert plain URLs to markdown links
    - Trim leading/trailing whitespace
    """
    if raw_text is None:
        return ""

    # If the response is a structure (dict-like), try to stringify sensibly
    if isinstance(raw_text, (dict, list)):
        try:
            raw_text = json.dumps(raw_text, indent=2)
        except Exception:
            raw_text = str(raw_text)

    text = str(raw_text)

    # Some LLM outputs include literal backslash-n sequences; unescape them
    # Use unicode_escape only if it looks like escaped sequences exist
    if "\\n" in text or "\\t" in text:
        try:
            text = text.encode("utf-8").decode("unicode_escape")
        except Exception:
            text = text.replace("\\n", "\n").replace("\\t", "\t")

    # Normalize CRLF to LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse many empty lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Convert URLs into markdown links (display-only)
    url_regex = re.compile(r"(https?://[^\s)\]]+)")

    def _linkify(match):
        url = match.group(0)
        # keep label short
        return f"[{url}]({url})"

    text = url_regex.sub(_linkify, text)

    return text.strip()


# ---------- Small UI CSS polish ----------
st.markdown(
    """
    <style>
    /* overall container */
    .chat-container {
        background: var(--surface);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(48,54,61,0.25);
        margin-bottom: 1rem;
    }

    .user-message {
        background: linear-gradient(90deg, rgba(131,158,101,0.08), rgba(131,158,101,0.04));
        border-left: 4px solid rgba(131,158,101,0.9);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .assistant-message {
        background: linear-gradient(90deg, rgba(48,54,61,0.04), rgba(48,54,61,0.02));
        border-left: 4px solid rgba(48,54,61,0.85);
        border-radius: 8px;
        padding: 0.75rem;
        margin-bottom: 0.5rem;
    }

    .msg-meta {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-bottom: 6px;
    }

    .plot-container {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(48,54,61,0.12);
        padding: 8px;
        margin-top: 8px;
        background: var(--surface);
    }

    /* input */
    .stTextInput>div>div>input {
        padding: 10px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Page Header ----------
st.markdown(
    """
    <div style='text-align:center; margin-bottom: 1rem;'>
        <h1 style='margin:0; font-size:38px;'>📈 Finance ChatBot</h1>
        <div style='color:var(--text-secondary); margin-top:6px;'>
            Meet your <strong>MoneyMentor</strong> — ask about markets, upload charts, or get financial insights.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Ensure temp folder for images -----------
os.makedirs("temp_images", exist_ok=True)

# ---------- Language selection (sidebar) ----------
def select_language():
    languages = {
        "English": "english",
        "Hindi": "hindi",
        "Telugu": "telugu",
        "Urdu": "urdu",
        "Tamil": "tamil",
        "Marathi": "marathi",
        "Bengali": "bengali",
        "Gujarati": "gujarati",
        "Punjabi": "punjabi",
        "Kannada": "kannada",
        "Malayalam": "malayalam",
        "Odia": "odia",
        "Assamese": "assamese",
    }
    st.sidebar.header("🌐 Language")
    selected = st.sidebar.selectbox("Choose language", list(languages.keys()), index=0)
    return languages[selected]

selected_language = select_language()

# ---------- Initialize chatbot instance in session ----------
if "chatbot" not in st.session_state:
    st.session_state.chatbot = FinancialChatBot(language=selected_language)

if "history" not in st.session_state:
    # History as list of alternating user/assistant entries:
    # user: {"role":"user", "content": "...", "image_path": None}
    # assistant: {"role":"assistant", "text":"...", "plot": <json str or None>}
    st.session_state.history = []

# ---------- Render existing conversation ----------
for i in range(0, len(st.session_state.history), 2):
    if i + 1 >= len(st.session_state.history):
        break
    user_msg = st.session_state.history[i]
    assistant_msg = st.session_state.history[i + 1]

    with st.container():
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        cols = st.columns([3, 2])

        # Left: messages (user + assistant)
        with cols[0]:
            # User block
            st.markdown(
                "<div class='user-message'>"
                f"<div class='msg-meta'>👤 Your Query</div>"
                f"{st.markdown(user_msg['content'], unsafe_allow_html=False) if False else ''}"
                "</div>",
                unsafe_allow_html=True,
            )
            # We render the user's content using st.write to get better escaping & formatting
            st.write(user_msg["content"])

            # Assistant block
            st.markdown("<div class='assistant-message'>", unsafe_allow_html=True)
            st.markdown("<div class='msg-meta'>🤖 Analysis</div>", unsafe_allow_html=True)

            # Clean assistant text and render as markdown
            cleaned = clean_assistant_text(assistant_msg.get("text", ""))
            # Use st.markdown to allow markdown formatting (headings, lists, links)
            st.markdown(cleaned, unsafe_allow_html=False)

            st.markdown("</div>", unsafe_allow_html=True)

        # Right: image / plot / resources
        with cols[1]:
            # image uploaded by user (if any)
            if user_msg.get("image_path"):
                try:
                    st.image(user_msg["image_path"], caption="Uploaded Chart (user)", use_container_width=True)
                except Exception:
                    pass

            # If assistant included a plot (JSON string), try to render
            if assistant_msg.get("plot"):
                try:
                    with st.container():
                        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                        plot_json = assistant_msg["plot"]
                        # plot_json might already be a dict or a json string
                        if isinstance(plot_json, str):
                            try:
                                plot_dict = json.loads(plot_json)
                                fig = go.Figure(plot_dict)
                            except Exception:
                                # fallback: try plotly.from_json
                                try:
                                    fig = from_json(plot_json)
                                except Exception:
                                    fig = None
                        elif isinstance(plot_json, dict):
                            fig = go.Figure(plot_json)
                        else:
                            fig = None

                        if fig is not None:
                            fig.update_layout(
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                margin=dict(l=0, r=0, t=30, b=0),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Plot is present but could not be rendered.")
                        st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error rendering plot: {e}")

            # resources links (extract urls from assistant message)
            resources = re.findall(r"https?://\S+", assistant_msg.get("text", ""))
            if resources:
                with st.expander("Explore Resources", expanded=False):
                    for url in resources:
                        st.markdown(f"- 🌐 [{url}]({url})")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Input form ----------
with st.form(key="chat_form", clear_on_submit=True):
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    input_cols = st.columns([9, 1])

    with input_cols[0]:
        user_input = st.text_input(
            "Your message:",
            placeholder="Ask about market trends, stock analysis, or upload a chart...",
            label_visibility="collapsed",
        )

    with input_cols[1]:
        submit_button = st.form_submit_button(label="🚀 Analyze")

    uploaded_file = st.file_uploader(
        "📤 Drop Financial Charts Here",
        type=["png", "jpg", "jpeg"],
        key="file_uploader",
        help="Upload charts for instant analysis",
        label_visibility="collapsed",
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- When user submits ----------
if submit_button and (user_input or uploaded_file):
    with st.spinner("Analyzing..."):
        image_path = None
        if uploaded_file is not None:
            timestamp = int(time.time())
            image_path = f"temp_images/image_{timestamp}.png"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            if not user_input:
                user_input = "Describe what you see in this image. Focus on charts, technical indicators, or any notable patterns."

        # Ensure chatbot language matches selection
        st.session_state.chatbot.language = selected_language

        # Call chatbot
        try:
            bot_response = st.session_state.chatbot.chat(user_input, image_path)
            bot_text = bot_response.get("text", "")
            bot_plot = bot_response.get("plot", None)
        except Exception as e:
            logger.exception("Chatbot call failed: %s", e)
            bot_text = f"Sorry, I couldn't process that due to internal error: {e}"
            bot_plot = None

        # Save into session history (user then assistant)
        st.session_state.history.append({
            "role": "user",
            "content": user_input,
            "image_path": image_path,
        })

        # Clean assistant text for nicer display & storage
        cleaned_text_for_storage = clean_assistant_text(bot_text)

        st.session_state.history.append({
            "role": "assistant",
            "text": cleaned_text_for_storage,
            "plot": bot_plot,
        })

        # Optional: persist to Firestore (if you want to persist each message)
        try:
            user_email = st.session_state.get("user_info", {}).get("email", "unknown")
            # Create chat doc if not present in session
            if "current_chat_id" not in st.session_state:
                st.session_state.current_chat_id = create_new_chat(user_email)
            # Save user message
            save_message(st.session_state.current_chat_id, user_input, "user")
            # Save assistant message
            save_message(st.session_state.current_chat_id, cleaned_text_for_storage, "assistant")
        except Exception:
            logger.exception("Error persisting chat to Firestore (non-fatal).")

        # re-run to show updated UI
        st.rerun()

