import streamlit as st
import uuid
from datetime import datetime
from database import get_connection, initialise_database
from nlp.intent import get_intent
from nlp.entities import extract_entities, get_crs_code, find_stations_in_text, resolve_london
from engine.engine import run_engine
from config import INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY, INTENT_ADD_RULE, INTENT_UNKNOWN
from engine.knowledge_base import get_kb

st.set_page_config(
    page_title="Train Assistant",
    page_icon="🚂",
    layout="centered"
)

if "db_ready" not in st.session_state:
    initialise_database()
    st.session_state.db_ready = True

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "bot",
        "content": (
            "Hello! I'm your train assistant 🚂\n\n"
            "I can help you with:\n"
            "- Finding the cheapest train ticket\n"
            "- Predicting your arrival time if your "
            "train is delayed\n\n"
            "How can I help you today?"
        )
    })

if "intent" not in st.session_state:
    st.session_state.intent = None

# stage tracks what we asked for last
# so we know how to interpret the next bare answer
if "stage" not in st.session_state:
    st.session_state.stage = None

if "collected" not in st.session_state:
    st.session_state.collected = {
        "origin": None, "destination": None,
        "origin_crs": None, "destination_crs": None,
        "date": None, "ticket_type": None,
        "current_station": None, "current_station_crs": None,
        "planned_arrival": None,
        "day_of_week": None, "month": None,
    }

# ─── Knowledge Acquisition state ────────────────────────
if "ka_category" not in st.session_state:
    st.session_state.ka_category = None
if "ka_question" not in st.session_state:
    st.session_state.ka_question = None
if "ka_keywords" not in st.session_state:
    st.session_state.ka_keywords = None
if "ka_answer" not in st.session_state:
    st.session_state.ka_answer = None


def log_to_db(user_msg, bot_msg, intent=None):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO conversation_history "
            "(timestamp, user_msg, bot_msg, intent) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now().isoformat(), user_msg,
             bot_msg, intent)
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def resolve_station(text: str, intent: str):
    """Try to find a station name and CRS in text."""
    stations = find_stations_in_text(text)
    if stations:
        return stations[0][0], stations[0][1]
    raw = text.strip().title()
    if raw.lower() == "london":
        name, crs = resolve_london("london", intent)
        return name, crs
    return raw, get_crs_code(raw)


def process_message(user_input: str) -> str:
    c = st.session_state.collected
    stage = st.session_state.stage

    # ── Step 1: KB check — if not mid-conversation, check
    #    the knowledge base first before doing intent routing.
    #    This prevents KB questions like "Do you offer group
    #    tickets?" being swallowed by the book_ticket intent.
    if stage is None:
        kb = get_kb()
        kb_result = kb.search_qa(user_input)
        # Only use KB answer if it's not a clear booking/delay
        # trigger — those should still go through the full flow
        booking_triggers = {"book", "find", "buy", "cheapest", "price", "fare"}
        delay_triggers = {"delay", "late", "arrival", "predict", "arrive"}
        words = set(user_input.lower().split())
        is_transactional = bool(words & booking_triggers) or bool(words & delay_triggers)
        if kb_result and not is_transactional:
            return kb_result["answer"]

    # ── Step 2: Detect intent ─────────────────────────────
    detected = get_intent(user_input)
    if detected != INTENT_UNKNOWN:
        st.session_state.intent = detected
    intent = st.session_state.intent or INTENT_UNKNOWN

    # ── Step 2b: Intent changed mid-flow — reset stage ────
    # If the user switched topics while we were mid-conversation
    # (e.g. typed "my train is delayed" while we were asking for
    # an origin station), reset everything and start fresh.
    if stage is not None and detected != INTENT_UNKNOWN:
        stage_intent_map = {
            "ask_origin": INTENT_BOOK_TICKET,
            "ask_destination": INTENT_BOOK_TICKET,
            "ask_date": INTENT_BOOK_TICKET,
            "ask_ticket_type": INTENT_BOOK_TICKET,
            "ask_current_station": INTENT_PREDICT_DELAY,
            "ask_delay_destination": INTENT_PREDICT_DELAY,
            "ask_planned_arrival": INTENT_PREDICT_DELAY,
            # ka_ask_* stages are intentionally excluded — in those stages
            # the user is typing question/keyword/answer content, not intent.
            # e.g. "Do you offer group tickets?" should NOT trigger BOOK_TICKET.
        }
        expected = stage_intent_map.get(stage)
        if expected and detected != expected:
            st.session_state.stage = None
            st.session_state.collected = {k: None for k in c}
            st.session_state.intent = detected
            stage = None
            intent = detected
            c = st.session_state.collected

    # ── Step 3: If we're mid-conversation, collect the
    #    answer to the question we just asked ─────────────
    if stage == "ask_origin":
        name, crs = resolve_station(user_input, intent)
        if not crs:
            return (f"Sorry, I couldn't find a station called '{name}'. "
                    "Please try again — for example: Norwich, Southampton, Oxford.")
        c["origin"] = name
        c["origin_crs"] = crs
        st.session_state.stage = None

    elif stage == "ask_destination":
        name, crs = resolve_station(user_input, intent)
        if not crs:
            return (f"Sorry, I couldn't find a station called '{name}'. "
                    "Please try again — for example: London, Oxford, Winchester.")
        c["destination"] = name
        c["destination_crs"] = crs
        st.session_state.stage = None

    elif stage == "ask_date":
        entities = extract_entities(user_input, intent)
        c["date"] = entities.get("date") or user_input.strip()
        st.session_state.stage = None

    elif stage == "ask_ticket_type":
        lower = user_input.lower()
        c["ticket_type"] = "return" if "return" in lower else "single"
        st.session_state.stage = None

    elif stage == "ask_current_station":
        name, crs = resolve_station(user_input, intent)
        if not crs:
            return (f"Sorry, I couldn't find '{name}' on the Weymouth–Waterloo line. "
                    "Try: Weymouth, Wareham, Poole, Bournemouth, Southampton, Winchester.")
        c["current_station"] = name
        c["current_station_crs"] = crs
        st.session_state.stage = None

    elif stage == "ask_delay_destination":
        name, crs = resolve_station(user_input, intent)
        if not crs:
            return (f"Sorry, I couldn't find '{name}'. "
                    "Try: London Waterloo, Winchester, Southampton.")
        c["destination"] = name
        c["destination_crs"] = crs
        st.session_state.stage = None

    elif stage == "ask_planned_arrival":
        entities = extract_entities(user_input, intent)
        c["planned_arrival"] = (
            entities.get("time") or
            entities.get("date") or
            user_input.strip()
        )
        now = datetime.now()
        c["day_of_week"] = now.weekday()
        c["month"] = now.month
        st.session_state.stage = None

    # ─── KA stages ───────────────────────────────────────
    elif stage == "ka_ask_category":
        st.session_state.ka_category = user_input.strip().lower().replace(" ", "_")
        st.session_state.stage = "ka_ask_question"
        return "What is the question or trigger phrase?"

    elif stage == "ka_ask_question":
        st.session_state.ka_question = user_input.strip()
        st.session_state.stage = "ka_ask_keywords"
        return ("What keywords should trigger this answer? "
                "Separate them with commas. For example: group ticket, bulk booking")

    elif stage == "ka_ask_keywords":
        st.session_state.ka_keywords = [
            k.strip() for k in user_input.split(",") if k.strip()
        ]
        st.session_state.stage = "ka_ask_answer"
        return "And what should the answer be?"

    elif stage == "ka_ask_answer":
        st.session_state.ka_answer = user_input.strip()
        # Persist to KB
        kb = get_kb()
        kb.add_qa(
            category=st.session_state.ka_category,
            question=st.session_state.ka_question,
            keywords=st.session_state.ka_keywords,
            answer=st.session_state.ka_answer,
        )
        summary = (
            f"✅ Learned! I've added this to the **{st.session_state.ka_category}** category:\n\n"
            f"**Q:** {st.session_state.ka_question}\n"
            f"**Keywords:** {', '.join(st.session_state.ka_keywords)}\n"
            f"**A:** {st.session_state.ka_answer}\n\n"
            "I'll use this to answer questions from now on."
        )
        # Reset KA state
        st.session_state.ka_category = None
        st.session_state.ka_question = None
        st.session_state.ka_keywords = None
        st.session_state.ka_answer = None
        st.session_state.stage = None
        st.session_state.intent = None
        return summary

    # ── Step 4: Decide what to ask or do next ────────────
    if intent == INTENT_BOOK_TICKET:
        if not c["origin"]:
            st.session_state.stage = "ask_origin"
            return ("Sure! I can help you find the cheapest "
                    "train ticket. Where are you travelling from?")
        if not c["destination"]:
            st.session_state.stage = "ask_destination"
            return "Great — and where are you travelling to?"
        if not c["date"]:
            st.session_state.stage = "ask_date"
            return ("What date are you planning to travel? "
                    "For example: 15th July.")
        if not c["ticket_type"]:
            st.session_state.stage = "ask_ticket_type"
            return "Would you like a single or return ticket?"

        # All info collected — search for ticket
        from task1.ticket_api import find_cheapest_ticket
        result = find_cheapest_ticket(
            origin_crs=c["origin_crs"] or "NRW",
            destination_crs=c["destination_crs"] or "LST",
            date_string=c["date"]
        )
        # Reset for next conversation
        st.session_state.intent = None
        st.session_state.stage = None
        st.session_state.collected = {
            k: None for k in c}
        if result["found"]:
            return (
                f"Great news! The cheapest ticket from "
                f"{c['origin']} to {c['destination']} "
                f"on {c['date']} is **{result['price']}** "
                f"({result['ticket_type']}).\n\n"
                f"🕐 Departs: "
                f"{result['departure'][:16].replace('T',' ')}\n"
                f"🏁 Arrives: "
                f"{result['arrival'][:16].replace('T',' ')}\n\n"
                f"[Book here]({result['booking_url']})"
            )
        return (
            f"I couldn't find a fare right now. "
            f"[Search directly here]({result['booking_url']})"
        )

    elif intent == INTENT_PREDICT_DELAY:
        if not c["current_station"]:
            st.session_state.stage = "ask_current_station"
            return ("I can help predict your arrival time. "
                    "Which station is your train at currently?")
        if not c["destination"]:
            st.session_state.stage = "ask_delay_destination"
            return "Where are you heading to?"
        if not c["planned_arrival"]:
            st.session_state.stage = "ask_planned_arrival"
            return ("What was the originally scheduled "
                    "arrival time at your destination? "
                    "For example: 11:30.")

        # All info collected — predict delay
        from task2.predict import predict_delay
        result = predict_delay(
            current_station_crs=c["current_station_crs"] or "SOU",
            destination_crs=c["destination_crs"] or "WAT",
            planned_arrival_time=c["planned_arrival"],
            planned_departure_time=c["planned_arrival"],
            direction="WEY2WAT",
            day_of_week=c["day_of_week"] or 0,
            month=c["month"] or 7
        )
        delay = result["predicted_delay_minutes"]
        arrival = result["predicted_arrival"]
        confidence = result["confidence"]

        if delay > 0:
            delay_text = f"approximately {abs(delay):.0f} minutes late"
        elif delay < 0:
            delay_text = f"approximately {abs(delay):.0f} minutes early"
        else:
            delay_text = "on time"

        # Reset
        st.session_state.intent = None
        st.session_state.stage = None
        st.session_state.collected = {k: None for k in c}

        return (
            f"Based on historical data, your train is predicted "
            f"to arrive at {c['destination']} **{delay_text}**.\n\n"
            f"🕐 Estimated arrival: **{arrival}**\n"
            f"📊 Confidence: {confidence}"
        )

    elif intent == INTENT_ADD_RULE:
        st.session_state.stage = "ka_ask_category"
        return ("Sure, I can learn something new! "
                "What category should this go in? "
                "For example: ticket_types, booking_policies, delays")

    else:
        return (
            "I'm sorry, I didn't quite understand that. "
            "I can help you find a train ticket, predict "
            "your arrival time, or you can teach me something "
            "new by saying 'add a rule'. Which would you like?"
        )


# ─── UI ──────────────────────────────────────────────────
st.title("🚂 Train Assistant Chatbot")
st.caption(f"Session: {st.session_state.session_id[:8]}...")

for msg in st.session_state.messages:
    role = "assistant" if msg["role"] == "bot"  else "user"
    with st.chat_message(role):
        st.write(msg["content"])

user_input = st.chat_input("Type your message here...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input})
    bot_response = process_message(user_input)
    st.session_state.messages.append(
        {"role": "bot", "content": bot_response})
    log_to_db(user_input, bot_response,
              st.session_state.intent)
    st.rerun()
