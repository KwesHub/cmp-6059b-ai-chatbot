import streamlit as st
import uuid
from datetime import datetime
from database import get_connection, initialise_database

# ─── Page config ────────────────────────────────────────
st.set_page_config(
    page_title="Train Assistant",
    page_icon="🚂",
    layout="centered"
)

# ─── Initialise database on startup ─────────────────────
# Only runs once — session_state prevents re-running on every Streamlit rerun
if "db_initialised" not in st.session_state:
    initialise_database()
    st.session_state.db_initialised = True

# ─── Session state setup ────────────────────────────────
# st.session_state persists values across reruns of the app
# Every time a user sends a message, Streamlit reruns the
# whole script — session_state keeps things from resetting

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add the opening greeting
    st.session_state.messages.append({
        "role": "bot",
        "content": "Hello! I'm your train assistant. I can help you find the cheapest train ticket or predict your arrival time if your train is delayed. How can I help you today?"
    })


# ─── Helper: log conversation to database ───────────────
def log_to_db(user_msg, bot_msg):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO conversation_history
            (timestamp, user_msg, bot_msg, intent)
        VALUES (?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        user_msg,
        bot_msg,
        None  # intent gets filled in later by the NLP module
    ))
    conn.commit()
    conn.close()


# ─── Helper: placeholder bot response ───────────────────
# This gets replaced in Milestone 5 when experta is wired in
def get_bot_response(user_input):
    return f"I received your message: '{user_input}'. The NLP and reasoning engine will be connected soon."


# ─── UI ─────────────────────────────────────────────────
st.title("🚂 Train Assistant Chatbot")
st.caption(f"Session: {st.session_state.session_id[:8]}...")

# Display message history
for msg in st.session_state.messages:
    if msg["role"] == "bot":
        with st.chat_message("assistant"):
            st.write(msg["content"])
    else:
        with st.chat_message("user"):
            st.write(msg["content"])

# Chat input box — appears at the bottom
user_input = st.chat_input("Type your message here...")

if user_input:
    # Add user message to display
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # Get bot response
    bot_response = get_bot_response(user_input)

    # Add bot response to display
    st.session_state.messages.append({
        "role": "bot",
        "content": bot_response
    })

    # Save to database
    log_to_db(user_input, bot_response)

    # Rerun to update the display
    st.rerun()