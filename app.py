import streamlit as st
import uuid
from datetime import datetime
from database import get_connection, initialise_database, seed_station_codes
from nlp.intent import get_intent
from nlp.entities import extract_entities, get_crs_code, find_stations_in_text, resolve_london, find_stations_fuzzy, find_station_by_typo
from config import INTENT_BOOK_TICKET, INTENT_PREDICT_DELAY, INTENT_ADD_RULE, INTENT_UNKNOWN
from engine.knowledge_base import get_kb

st.set_page_config(
    page_title="Train Assistant",
    page_icon="🚂",
    layout="centered"
)

if "db_ready" not in st.session_state:
    initialise_database()
    seed_station_codes()   # loads ALL stations from CSV
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
        "date": None, "depart_time": None, "ticket_type": None,
        "return_date": None, "return_time": None,
        "current_station": None, "current_station_crs": None,
        "planned_arrival": None, "delay_minutes": None,
        "day_of_week": None, "month": None,
    }

# ─── Knowledge Acquisition state ────────────────────────
if "_confirm_prefix" not in st.session_state:
    st.session_state._confirm_prefix = None
if "typo_suggestion" not in st.session_state:
    st.session_state.typo_suggestion = None   # (name, crs) pending typo confirmation
if "typo_stage_return" not in st.session_state:
    st.session_state.typo_stage_return = None # stage to resume after typo confirm
if "parsed_date" not in st.session_state:
    st.session_state.parsed_date = None       # ISO date string pending confirmation

# ─── Disambiguation state ────────────────────────────────
if "disambig_options" not in st.session_state:
    st.session_state.disambig_options = []   # list of (name, crs)
if "disambig_for" not in st.session_state:
    st.session_state.disambig_for = None     # "origin","destination","current_station","delay_dest"

# ─── Knowledge Acquisition state ─────────────────────────
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
    """
    Try to find a station. Returns (name, crs) on exact match,
    or ("__ambiguous__", options_list) when multiple matches found,
    or (raw_text, None) when nothing found.
    """
    text = text.strip()

    # 0. Handle "London" → route to correct terminus before fuzzy matching
    london_name, london_crs = resolve_london(text, intent)
    if london_crs:
        return london_name, london_crs

    # 1. Try exact / substring match in DB
    stations = find_stations_in_text(text)
    if len(stations) == 1:
        return stations[0][0], stations[0][1]
    if len(stations) > 1:
        # Multiple exact matches — let user pick
        return "__ambiguous__", stations[:3]

    # 2. Fuzzy match
    fuzzy = find_stations_fuzzy(text, limit=3)
    if len(fuzzy) == 1:
        return fuzzy[0][0], fuzzy[0][1]
    if len(fuzzy) > 1:
        return "__ambiguous__", fuzzy

    # 3. Title-case direct lookup
    raw = text.title()
    crs = get_crs_code(raw)
    if crs:
        return raw, crs

    # 4. Typo / close-match suggestion
    typo_name, typo_crs = find_station_by_typo(text)
    if typo_name:
        return "__typo__", (typo_name, typo_crs)

    return raw, None


def build_disambig_prompt(options: list, context: str) -> str:
    """Format a 'which station did you mean?' message."""
    lines = "\n".join(
        f"  {i+1}. {name}" for i, (name, _) in enumerate(options)
    )
    return (
        f"I found a few stations matching '{context}':\n\n"
        f"{lines}\n\n"
        "Which one do you mean? Reply with the number."
    )


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
            "confirm_date": INTENT_BOOK_TICKET,
            "ask_time": INTENT_BOOK_TICKET,
            "ask_ticket_type": INTENT_BOOK_TICKET,
            "ask_return_date": INTENT_BOOK_TICKET,
            "confirm_return_date": INTENT_BOOK_TICKET,
            "ask_return_time": INTENT_BOOK_TICKET,
            "disambiguate_station": None,  # handled by disambig_for context
            "confirm_typo": None,          # protected — user is answering yes/no
            "ask_current_station": INTENT_PREDICT_DELAY,
            "ask_delay_destination": INTENT_PREDICT_DELAY,
            "ask_planned_arrival": INTENT_PREDICT_DELAY,
            "ask_delay_minutes": INTENT_PREDICT_DELAY,
            # ka_ask_* excluded — user types content, not intent
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

    # ── Disambiguation: user is picking from a numbered list ─
    if stage == "disambiguate_station":
        options = st.session_state.disambig_options
        disambig_for = st.session_state.disambig_for
        choice = user_input.strip()
        picked = None

        # Accept a number
        if choice in ("1", "2", "3") and int(choice) <= len(options):
            picked = options[int(choice) - 1]
        else:
            # Try matching against the listed options
            for opt in options:
                if choice.lower() in opt[0].lower():
                    picked = opt
                    break

        if not picked:
            # User typed something entirely different — treat as a new station query
            new_name, new_crs = resolve_station(user_input, intent)
            if new_name not in ("__ambiguous__", "__typo__") and new_crs:
                # They typed a valid new station — accept it
                picked = (new_name, new_crs)
            elif new_name == "__ambiguous__":
                # New ambiguous query — start fresh disambiguation
                st.session_state.disambig_options = new_crs
                st.session_state.stage = "disambiguate_station"
                return build_disambig_prompt(new_crs, user_input.strip())
            else:
                # Still can't find it — show options again
                lines = "\n".join(f"  {i+1}. {n}" for i, (n, _) in enumerate(options))
                return (f"I didn't recognise that. Please reply with a number, "
                        f"or type a different station name:\n\n{lines}")
        name, crs = picked
        # Store in the right field
        if disambig_for == "origin":
            c["origin"] = name; c["origin_crs"] = crs
            confirm = f"✅ Travelling from **{name}**."
        elif disambig_for == "destination":
            c["destination"] = name; c["destination_crs"] = crs
            confirm = f"✅ Destination: **{name}**."
        elif disambig_for == "current_station":
            c["current_station"] = name; c["current_station_crs"] = crs
            confirm = f"✅ Current station: **{name}**."
        elif disambig_for == "delay_dest":
            c["destination"] = name; c["destination_crs"] = crs
            confirm = f"✅ Heading to **{name}**."
        else:
            confirm = f"✅ Got it — **{name}**."
        st.session_state.disambig_options = []
        st.session_state.disambig_for = None
        st.session_state.stage = None
        # Prepend confirm to whatever we ask next (fall through to Step 4)
        st.session_state._confirm_prefix = confirm

    elif stage == "confirm_typo":
        # User is confirming/rejecting a typo suggestion
        answer = user_input.strip().lower()
        if answer in ("yes", "y", "yeah", "yep", "correct", "ok", "sure"):
            name, crs = st.session_state.typo_suggestion
            return_stage = st.session_state.typo_stage_return
            st.session_state.typo_suggestion = None
            st.session_state.typo_stage_return = None
            st.session_state.stage = None
            if return_stage == "ask_origin":
                c["origin"] = name; c["origin_crs"] = crs
                st.session_state._confirm_prefix = f"✅ Travelling from **{name}**."
            elif return_stage == "ask_destination":
                c["destination"] = name; c["destination_crs"] = crs
                st.session_state._confirm_prefix = f"✅ Destination: **{name}**."
            elif return_stage == "ask_current_station":
                c["current_station"] = name; c["current_station_crs"] = crs
                st.session_state._confirm_prefix = f"✅ Current station: **{name}**."
            elif return_stage == "ask_delay_destination":
                c["destination"] = name; c["destination_crs"] = crs
                st.session_state._confirm_prefix = f"✅ Heading to **{name}**."
        else:
            # Rejected — go back to asking for the station
            st.session_state.typo_suggestion = None
            st.session_state.stage = st.session_state.typo_stage_return
            st.session_state.typo_stage_return = None
            return "No problem — please try again with the station name."

    elif stage == "confirm_date":
        # User is confirming the parsed date shown to them
        answer = user_input.strip().lower()
        if answer in ("yes", "y", "yeah", "yep", "correct", "ok", "sure", "that's right", "thats right"):
            c["date"] = st.session_state.parsed_date
            st.session_state.parsed_date = None
            st.session_state.stage = None
            st.session_state._confirm_prefix = f"✅ Travel date confirmed."
        else:
            # They said no or typed a correction — treat as new date
            st.session_state.parsed_date = None
            st.session_state.stage = "ask_date"
            c["date"] = None
            return "No problem — what date would you like to travel? For example: 15th July or tomorrow."

    elif stage == "ask_origin":
        name, crs = resolve_station(user_input, intent)
        if name == "__ambiguous__":
            st.session_state.disambig_options = crs
            st.session_state.disambig_for = "origin"
            st.session_state.stage = "disambiguate_station"
            return build_disambig_prompt(crs, user_input.strip())
        if name == "__typo__":
            sug_name, sug_crs = crs  # crs holds (name, crs) here
            st.session_state.typo_suggestion = (sug_name, sug_crs)
            st.session_state.typo_stage_return = "ask_origin"
            st.session_state.stage = "confirm_typo"
            return f"Did you mean **{sug_name}**? (yes / no)"
        if not crs:
            return (f"Sorry, I couldn't find a station called '{name}'. "
                    "Try again — for example: Norwich, Southampton, Oxford.")
        c["origin"] = name
        c["origin_crs"] = crs
        st.session_state._confirm_prefix = f"✅ Travelling from **{name}**."
        st.session_state.stage = None

    elif stage == "ask_destination":
        name, crs = resolve_station(user_input, intent)
        if name == "__ambiguous__":
            st.session_state.disambig_options = crs
            st.session_state.disambig_for = "destination"
            st.session_state.stage = "disambiguate_station"
            return build_disambig_prompt(crs, user_input.strip())
        if name == "__typo__":
            sug_name, sug_crs = crs
            st.session_state.typo_suggestion = (sug_name, sug_crs)
            st.session_state.typo_stage_return = "ask_destination"
            st.session_state.stage = "confirm_typo"
            return f"Did you mean **{sug_name}**? (yes / no)"
        if not crs:
            return (f"Sorry, I couldn't find a station called '{name}'. "
                    "Try again — for example: London, Oxford, Winchester.")
        c["destination"] = name
        c["destination_crs"] = crs
        st.session_state._confirm_prefix = f"✅ Destination: **{name}**."
        st.session_state.stage = None

    elif stage == "ask_date":
        from task1.ticket_api import format_datetime
        raw_input = user_input.strip()
        parsed_iso = format_datetime(raw_input, hour=9)
        # Show the parsed date in readable format and ask to confirm
        try:
            from datetime import datetime as _dt
            parsed_dt = _dt.strptime(parsed_iso, "%Y-%m-%dT%H:%M:%S")
            readable = parsed_dt.strftime("%-d %B %Y")  # e.g. "8 May 2026"
            if parsed_dt.hour != 9 or parsed_dt.minute != 0:
                readable += parsed_dt.strftime(" at %H:%M")
        except Exception:
            readable = raw_input
        st.session_state.parsed_date = parsed_iso
        st.session_state.stage = "confirm_date"
        return f"Just to confirm — you'd like to travel on **{readable}**. Is that right? (yes / no)"

    elif stage == "ask_ticket_type":
        lower = user_input.lower()
        c["ticket_type"] = "return" if "return" in lower else "single"
        st.session_state._confirm_prefix = f"✅ Ticket type: **{c['ticket_type']}**."
        st.session_state.stage = None

    elif stage == "ask_time":
        import re as _re
        raw_time = user_input.strip().lower()
        # "any", "skip", "no preference", blank → search from first train of the day
        if raw_time in ("any", "skip", "no", "none", "no preference", "doesn't matter", ""):
            c["depart_time"] = "any"
            st.session_state._confirm_prefix = "✅ Searching all day for the cheapest fare."
        else:
            # Try to extract HH:MM from input (e.g. "9pm", "14:30", "9 am")
            match = _re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', raw_time)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                meridiem = match.group(3)
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                c["depart_time"] = f"{hour:02d}:{minute:02d}"
                st.session_state._confirm_prefix = f"✅ Departure time: **{c['depart_time']}**."
            else:
                c["depart_time"] = "09:00"
                st.session_state._confirm_prefix = "✅ Couldn't parse that time — defaulting to 09:00."
        st.session_state.stage = None

    elif stage == "ask_return_date":
        from task1.ticket_api import format_datetime
        raw_input = user_input.strip()
        parsed_iso = format_datetime(raw_input, hour=9)
        try:
            from datetime import datetime as _dt
            parsed_dt = _dt.strptime(parsed_iso, "%Y-%m-%dT%H:%M:%S")
            readable = parsed_dt.strftime("%-d %B %Y")
        except Exception:
            readable = raw_input
        st.session_state.parsed_date = parsed_iso
        st.session_state.stage = "confirm_return_date"
        return f"Just to confirm — you'd like to return on **{readable}**. Is that right? (yes / no)"

    elif stage == "confirm_return_date":
        answer = user_input.strip().lower()
        if answer in ("yes", "y", "yeah", "yep", "correct", "ok", "sure", "that's right", "thats right"):
            c["return_date"] = st.session_state.parsed_date
            st.session_state.parsed_date = None
            st.session_state.stage = None
            st.session_state._confirm_prefix = "✅ Return date confirmed."
        else:
            st.session_state.parsed_date = None
            st.session_state.stage = "ask_return_date"
            c["return_date"] = None
            return "No problem — what date would you like to return?"

    elif stage == "ask_return_time":
        import re as _re
        raw_time = user_input.strip().lower()
        if raw_time in ("any", "skip", "no", "none", "no preference", "doesn't matter", ""):
            c["return_time"] = "any"
            st.session_state._confirm_prefix = "✅ Searching all day for cheapest return fare."
        else:
            match = _re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', raw_time)
            if match:
                hour = int(match.group(1))
                minute = int(match.group(2) or 0)
                meridiem = match.group(3)
                if meridiem == "pm" and hour != 12:
                    hour += 12
                elif meridiem == "am" and hour == 12:
                    hour = 0
                c["return_time"] = f"{hour:02d}:{minute:02d}"
                st.session_state._confirm_prefix = f"✅ Return time: **{c['return_time']}**."
            else:
                c["return_time"] = "14:00"
                st.session_state._confirm_prefix = "✅ Couldn't parse that — defaulting to 14:00."
        st.session_state.stage = None

    elif stage == "ask_current_station":
        name, crs = resolve_station(user_input, intent)
        if name == "__ambiguous__":
            st.session_state.disambig_options = crs
            st.session_state.disambig_for = "current_station"
            st.session_state.stage = "disambiguate_station"
            return build_disambig_prompt(crs, user_input.strip())
        if name == "__typo__":
            sug_name, sug_crs = crs
            st.session_state.typo_suggestion = (sug_name, sug_crs)
            st.session_state.typo_stage_return = "ask_current_station"
            st.session_state.stage = "confirm_typo"
            return f"Did you mean **{sug_name}**? (yes / no)"
        if not crs:
            return (f"Sorry, I couldn't find '{name}'. "
                    "Try: Weymouth, Wareham, Poole, Bournemouth, Southampton, Winchester.")
        c["current_station"] = name
        c["current_station_crs"] = crs
        st.session_state._confirm_prefix = f"✅ Current station: **{name}**."
        st.session_state.stage = None

    elif stage == "ask_delay_destination":
        name, crs = resolve_station(user_input, intent)
        if name == "__ambiguous__":
            st.session_state.disambig_options = crs
            st.session_state.disambig_for = "delay_dest"
            st.session_state.stage = "disambiguate_station"
            return build_disambig_prompt(crs, user_input.strip())
        if name == "__typo__":
            sug_name, sug_crs = crs
            st.session_state.typo_suggestion = (sug_name, sug_crs)
            st.session_state.typo_stage_return = "ask_delay_destination"
            st.session_state.stage = "confirm_typo"
            return f"Did you mean **{sug_name}**? (yes / no)"
        if not crs:
            return (f"Sorry, I couldn't find '{name}'. "
                    "Try: London Waterloo, Winchester, Southampton.")
        c["destination"] = name
        c["destination_crs"] = crs
        st.session_state._confirm_prefix = f"✅ Heading to **{name}**."
        st.session_state.stage = None

    elif stage == "ask_planned_arrival":
        c["planned_arrival"] = user_input.strip()
        now = datetime.now()
        c["day_of_week"] = now.weekday()
        c["month"] = now.month
        st.session_state._confirm_prefix = f"✅ Scheduled arrival: **{c['planned_arrival']}**."
        st.session_state.stage = None

    elif stage == "ask_delay_minutes":
        import re as _re
        nums = _re.findall(r'\d+', user_input)
        mins = int(nums[0]) if nums else 0
        c["delay_minutes"] = mins
        st.session_state._confirm_prefix = f"✅ Current delay: **{mins} minutes**."
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
    # Prepend any verbal confirmation from Step 3
    def _reply(msg: str) -> str:
        prefix = st.session_state.get("_confirm_prefix")
        st.session_state._confirm_prefix = None
        return f"{prefix}\n\n{msg}" if prefix else msg

    if intent == INTENT_BOOK_TICKET:
        if not c["origin"]:
            st.session_state.stage = "ask_origin"
            return _reply("Sure! I can help you find the cheapest "
                          "train ticket. Where are you travelling from?")
        if not c["destination"]:
            st.session_state.stage = "ask_destination"
            return _reply("And where are you travelling to?")
        if not c["date"]:
            st.session_state.stage = "ask_date"
            return _reply("What date are you planning to travel? "
                          "For example: 15th July or tomorrow.")
        if not c["ticket_type"]:
            st.session_state.stage = "ask_ticket_type"
            return _reply("Would you like a single or return ticket?")
        if c["depart_time"] is None:
            st.session_state.stage = "ask_time"
            return _reply("What time would you like to depart? "
                          "For example: 9am, 14:30. "
                          "Or type **any** to find the cheapest fare of the day.")
        if c["ticket_type"] == "return" and not c["return_date"]:
            st.session_state.stage = "ask_return_date"
            return _reply("What date would you like to return?")
        if c["ticket_type"] == "return" and c["return_time"] is None:
            st.session_state.stage = "ask_return_time"
            return _reply("What time would you like to depart on your return journey? "
                          "Or type **any** for the cheapest return fare.")

        # All info collected — search for ticket
        from task1.ticket_api import find_cheapest_ticket, format_datetime
        # Merge confirmed date with chosen departure time
        base_iso = c["date"] if "T" in str(c["date"]) else format_datetime(c["date"])
        depart_hhmm = c.get("depart_time") or "09:00"
        use_first_train = (depart_hhmm == "any")
        # firstTrainOfDay → use 06:00 in the booking link (reasonable morning time)
        link_hhmm = "06:00" if use_first_train else depart_hhmm
        travel_iso = base_iso[:10] + "T" + ("00:00" if use_first_train else depart_hhmm) + ":00"
        link_iso   = base_iso[:10] + "T" + link_hhmm + ":00"
        from task1.ticket_api import _nr_booking_url
        _ticket_type = c.get("ticket_type") or "single"
        _origin_crs  = c["origin_crs"] or "NRW"
        _dest_crs    = c["destination_crs"] or "LST"
        # Pre-build the booking URL with a sensible time (not 00:00)
        override_url = _nr_booking_url(
            _origin_crs, _dest_crs, link_iso, _ticket_type,
            origin_name=c.get("origin"),
            destination_name=c.get("destination")
        )
        # Build return datetime if needed
        if _ticket_type == "return" and c.get("return_date"):
            ret_base = c["return_date"] if "T" in str(c["return_date"]) else format_datetime(c["return_date"])
            ret_hhmm = c.get("return_time") or "14:00"
            use_first_return = (ret_hhmm == "any")
            return_iso = ret_base[:10] + "T" + ("00:00" if use_first_return else ret_hhmm) + ":00"
        else:
            return_iso = None

        result = find_cheapest_ticket(
            origin_crs=_origin_crs,
            destination_crs=_dest_crs,
            date_string=travel_iso,
            return_date=return_iso,
            origin_name=c.get("origin"),
            destination_name=c.get("destination"),
            ticket_type=_ticket_type,
            use_first_train=use_first_train
        )
        # Always use the pre-built URL (correct CRS, sensible time)
        result["booking_url"] = override_url
        # Consume any pending confirmation prefix (e.g. "✅ Return time: 14:00.")
        # so it appears above the fare result
        pending_prefix = st.session_state.get("_confirm_prefix")
        st.session_state._confirm_prefix = None
        # Reset for next conversation
        st.session_state.intent = None
        st.session_state.stage = None
        st.session_state.collected = {
            k: None for k in c}
        if result["found"]:
            try:
                _travel_dt = datetime.strptime(travel_iso[:10], "%Y-%m-%d")
                display_date = _travel_dt.strftime("%-d %B %Y")
            except Exception:
                display_date = c['date']
            depart_time = result['departure'][:16].replace('T', ' ')
            arrive_time = result['arrival'][:16].replace('T', ' ')

            if _ticket_type == "return" and c.get("return_date"):
                # Show outward + return breakdown
                try:
                    _ret_dt = datetime.strptime(return_iso[:10], "%Y-%m-%d")
                    display_return_date = _ret_dt.strftime("%-d %B %Y")
                except Exception:
                    display_return_date = c.get("return_date", "")
                ret_depart = result.get("return_departure", "N/A")[:16].replace('T', ' ')
                ret_arrive  = result.get("return_arrival",  "N/A")[:16].replace('T', ' ')
                out_price = result.get("outward_price", "?")
                in_price  = result.get("inward_price", "?")
                msg = (
                    f"Great news! Here are the cheapest fares for your return trip "
                    f"**{c['origin']}** ↔ **{c['destination']}**:\n\n"
                    f"**Outward** ({display_date}): **{out_price}** ({result['ticket_type'].split(' + ')[0]})\n"
                    f"🕐 Departs: {depart_time} → Arrives: {arrive_time}\n\n"
                    f"**Return** ({display_return_date}): **{in_price}** ({result['ticket_type'].split(' + ')[-1]})\n"
                    f"🕐 Departs: {ret_depart} → Arrives: {ret_arrive}\n\n"
                    f"**Total: {result['price']}**\n\n"
                    f"[Book on National Rail →]({result['booking_url']})\n\n"
                    f"*Search outward:* **{c['origin']}** → **{c['destination']}**, "
                    f"**{display_date}**\n"
                    f"*Search return:* **{c['destination']}** → **{c['origin']}**, "
                    f"**{display_return_date}**"
                )
            else:
                msg = (
                    f"Great news! The cheapest ticket from "
                    f"**{c['origin']}** to **{c['destination']}** "
                    f"on **{display_date}** is **{result['price']}** "
                    f"({result['ticket_type']}).\n\n"
                    f"🕐 Departs: {depart_time}\n"
                    f"🏁 Arrives: {arrive_time}\n\n"
                    f"[Book on National Rail →]({result['booking_url']})\n\n"
                    f"*On the booking page, search:* "
                    f"**{c['origin']}** → **{c['destination']}**, "
                    f"**{display_date}**, **{depart_time[-5:]}**"
                )
            return f"{pending_prefix}\n\n{msg}" if pending_prefix else msg
        error_msg = result.get("error", "No fares available")
        if "timed out" in error_msg.lower():
            reason = "The National Rail API timed out."
        elif "HTTP" in error_msg:
            reason = "The National Rail API is temporarily unavailable."
        elif "No fares" in error_msg:
            reason = f"No fares were found for that route on {c.get('date', 'that date')}."
        else:
            reason = "The ticket search couldn't be completed right now."
        return (
            f"⚠️ {reason}\n\n"
            f"You can still search and book directly on the National Rail website:\n\n"
            f"[Search for tickets →]({result['booking_url']})"
        )

    elif intent == INTENT_PREDICT_DELAY:
        if not c["current_station"]:
            st.session_state.stage = "ask_current_station"
            return _reply("I can help predict your arrival time. "
                          "Which station is your train currently at?")
        if not c["destination"]:
            st.session_state.stage = "ask_delay_destination"
            return _reply("And where are you heading to?")
        if not c["planned_arrival"]:
            st.session_state.stage = "ask_planned_arrival"
            return _reply("What is the scheduled arrival time at your destination? "
                          "For example: 11:30.")
        if c["delay_minutes"] is None:
            st.session_state.stage = "ask_delay_minutes"
            return _reply("How many minutes has the train been delayed so far? "
                          "For example: 10.")

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
        # The model predicts total delay at destination.
        # We also show the current known delay for context.
        current_delay = c["delay_minutes"] or 0
        predicted_delay = result["predicted_delay_minutes"]
        arrival = result["predicted_arrival"]
        confidence = result["confidence"]

        def _mins(n):
            n = int(round(abs(n)))
            return f"{n} minute" if n == 1 else f"{n} minutes"

        if predicted_delay > 0:
            delay_text = f"approximately **{_mins(predicted_delay)} late**"
        elif predicted_delay < 0:
            delay_text = f"approximately **{_mins(predicted_delay)} early**"
        else:
            delay_text = "**on time**"

        # Reset
        st.session_state.intent = None
        st.session_state.stage = None
        st.session_state.collected = {k: None for k in c}

        return (
            f"Your train is currently **{current_delay} minutes delayed** "
            f"at {c['current_station']}.\n\n"
            f"Based on historical data for this route, your train is predicted "
            f"to arrive at {c['destination']} {delay_text}.\n\n"
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
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ── Base ── */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f7fa;
}

/* ── Header / title area ── */
h1 {
    color: #1a1a2e !important;
    font-weight: 600 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.3px;
}
.stCaption { color: #9aa5b4 !important; font-size: 0.75rem !important; }

/* ── Chat container background ── */
[data-testid="stChatMessageContainer"] {
    background: transparent;
}

/* ── All message bubbles ── */
[data-testid="stChatMessage"] {
    background: #ffffff;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 8px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    border: 1px solid #e8ecf0;
}

/* ── Bot messages — slight blue tint on left border ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid #1a56db;
    background: #ffffff;
}

/* ── User messages — slightly grey ── */
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background: #f0f4ff;
    border-left: 3px solid #6b7adb;
}

/* ── Message text ── */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #1a1a2e !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
}

/* ── Links ── */
[data-testid="stChatMessage"] a {
    color: #1a56db !important;
    font-weight: 500;
    text-decoration: underline;
}

/* ── Input box — dark bg, white text ── */
[data-testid="stChatInput"] {
    background: #1a1a2e !important;
    border: 1.5px solid #2e3a5c !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #1a56db !important;
    box-shadow: 0 0 0 3px rgba(26,86,219,0.25) !important;
}
[data-testid="stChatInput"] textarea {
    color: #f0f4f8 !important;
    background: #1a1a2e !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: #6b7aad !important;
}
/* send button */
[data-testid="stChatInput"] button {
    background: #1a56db !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] button svg {
    fill: #ffffff !important;
}

/* ── Strong/bold in messages ── */
[data-testid="stChatMessage"] strong {
    color: #1a1a2e !important;
    font-weight: 600;
}

/* ── Main page background ── */
section.main > div {
    background-color: #f5f7fa;
}
</style>
""", unsafe_allow_html=True)

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
