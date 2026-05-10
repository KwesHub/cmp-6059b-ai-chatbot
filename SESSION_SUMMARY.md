# CMP-6059B Train Chatbot — Full Session Summary

## Project Overview

UEA Advanced AI coursework. Streamlit chatbot with two main features:
- **Task 1**: Find cheapest train ticket (National Rail OJP SOAP API)
- **Task 2**: Predict arrival delay (Keras ML model, Weymouth→Waterloo route)

**Tech stack**: Python, Streamlit, SQLite, spaCy (`en_core_web_sm`), scikit-learn, Keras/TensorFlow, experta (expert system), dateparser

---

## File Structure

```
cmp-6059b-ai-chatbot/
├── app.py                        # Main Streamlit app, all dialogue logic
├── config.py                     # DB_PATH, SPACY_MODEL, intent constants
├── database.py                   # SQLite schema + seed_station_codes()
├── requirements.txt
├── data/
│   └── StationNameAndCode.csv    # ~2,600 UK stations (name + CRS code)
├── nlp/
│   ├── intent.py                 # Intent classifier
│   └── entities.py               # Station/date/time entity extraction
├── task1/
│   └── ticket_api.py             # OJP SOAP API calls + fare parsing
├── task2/
│   ├── keras_model.py            # Train/save Keras delay prediction model
│   └── predict.py                # Load model + predict delay
└── engine/
    ├── knowledge_base.py         # KB: Q&A, rules, fallbacks (SQLite-backed)
    └── engine.py                 # experta expert system rules
```

**Note**: `chatbot.db` and trained model files are gitignored. After any code change, delete `chatbot.db` and restart Streamlit. Run `python task2/keras_model.py` on the demo machine to retrain.

---

## Intents

Defined in `config.py`:

| Constant | Value | Trigger |
|---|---|---|
| `INTENT_BOOK_TICKET` | `"book_ticket"` | "find ticket", "cheapest", "fare" etc. |
| `INTENT_PREDICT_DELAY` | `"predict_delay"` | "delay", "late", "arrival" etc. |
| `INTENT_ADD_RULE` | `"add_rule"` | "add a rule", "teach you" etc. |
| `INTENT_UNKNOWN` | `"unknown"` | fallback |

---

## Dialogue Flow (`app.py`)

### Session State Variables

```python
st.session_state.intent          # current intent
st.session_state.stage           # what we last asked (e.g. "ask_origin")
st.session_state.collected       # dict of all gathered info
st.session_state._confirm_prefix # prepended to next bot message ("✅ Travelling from Norwich.")
st.session_state.disambig_options  # list of (name, crs) for disambiguation
st.session_state.disambig_for      # "origin" | "destination" | "current_station" | "delay_dest"
st.session_state.typo_suggestion   # (name, crs) pending typo confirmation
st.session_state.typo_stage_return # stage to resume after typo confirm
st.session_state.parsed_date       # ISO date string pending confirmation
```

### `collected` dict keys

```python
{
    "origin": None, "destination": None,
    "origin_crs": None, "destination_crs": None,
    "date": None, "depart_time": None, "ticket_type": None,
    "current_station": None, "current_station_crs": None,
    "planned_arrival": None, "day_of_week": None, "month": None,
}
```

### Book Ticket Flow

1. Ask origin → `ask_origin` stage
2. Ask destination → `ask_destination`
3. Ask date → `ask_date` → parse → show readable date → `confirm_date` stage → confirm yes/no
4. Ask ticket type (single/return) → `ask_ticket_type`
5. Ask departure time or "any" → `ask_time`
6. All collected → call `find_cheapest_ticket()` → display result + booking link

When user picks **"any"** for time: `depart_time = "any"`, `use_first_train=True`, SOAP request uses `<ns:firstTrainOfDay>{date}</ns:firstTrainOfDay>`.

### Stage Flow for Station Input

Each station input goes through `resolve_station()`:

1. `find_stations_in_text()` — exact/substring DB match
2. `find_stations_fuzzy()` — starts-with + word-boundary contains
3. `get_crs_code()` — direct title-case lookup
4. `find_station_by_typo()` — difflib close match (cutoff 0.6)

Returns:
- `(name, crs)` — exact match found
- `("__ambiguous__", [(name, crs), ...])` — multiple matches → disambiguation stage
- `("__typo__", (name, crs))` — close match suggestion → confirm_typo stage
- `(raw_text, None)` — not found → error message

### Disambiguation

Shows numbered list, user replies 1/2/3. If user types a new name, `resolve_station` is called again. Stored in `disambig_options` + `disambig_for`.

### Verbal Confirmations

After each piece of info is collected, `_confirm_prefix` is set (e.g. `"✅ Travelling from Norwich."`). The `_reply()` helper prepends this to the next question. Clears itself after use.

### Intent Switch Protection

`stage_intent_map` defines which intent each stage belongs to. If detected intent ≠ expected intent while mid-flow, the flow resets. `ka_ask_*` stages are excluded (user types content not intent triggers). `confirm_typo` is also excluded.

---

## Station Resolution (`nlp/entities.py`)

### `find_stations_in_text(user_input)`

Three passes:
1. **Exact match** — `station_name.lower() == text_lower` → return immediately (prevents spurious multi-match)
2. **Station name ⊂ text** — sentence input, e.g. "travel from Norwich to Sheffield" → finds both
3. **Text ⊂ station name** — prefix/partial, e.g. "Southampton" → "Southampton Central" (min 4 chars)

### `find_stations_fuzzy(query, limit=3)`

- Exact match first (early return)
- Starts-with matches
- Contains matches using `re.search(r'\b' + re.escape(q), nl)` — word boundary prevents "london" matching "Caldon Low"
- Filters non-rail stops via `_is_likely_rail_station()`

### Non-rail filter

```python
_NON_RAIL_KEYWORDS = (
    "(bus)", "arena", "docks", "ferry", "(tramlink)",
    "tram", "metro", "underground", "tube"
)
```

### London disambiguation

```python
def resolve_london(station_name, intent):
    if station_name.lower() == "london":
        if intent == INTENT_PREDICT_DELAY:
            return ("London Waterloo", "WAT")
        else:
            return ("London Liverpool Street", "LST")
```

---

## Task 1: Ticket API (`task1/ticket_api.py`)

### OJP SOAP API

- **Endpoint**: `https://ojp.nationalrail.co.uk/webservices`
- **Auth**: HTTP Basic (env vars `NATIONAL_RAIL_USERNAME`, `NATIONAL_RAIL_PASSWORD`)
- **WSDL**: `https://ojp.nationalrail.co.uk/webservices?wsdl`
- **Operation**: `RealtimeJourneyPlan`
- **Namespaces**:
  - `NS_JP = "http://www.thalesgroup.com/ojp/jpservices"` — journey/fare elements
  - `NS_COM = "http://www.thalesgroup.com/ojp/common"` — common types incl. `Fare`

### SOAP Request

```xml
<ns:outwardTime>
    <ns:departBy>2026-07-15T09:00:00</ns:departBy>
    <!-- OR for "any time": -->
    <ns:firstTrainOfDay>2026-07-15</ns:firstTrainOfDay>
</ns:outwardTime>
```

`firstTrainOfDay` takes `xsd:date` (date only, no time). `departBy` takes `xsd:dateTime`.

### Response Parsing

The parser tries **NS_JP, NS_COM, and no-namespace** for every element — the API's namespace usage is inconsistent. Key elements:

- `outwardJourney` (NS_JP) → each journey
- `fare` (NS_JP) → each fare within journey
- `totalPrice` (NS_COM, PriceInPence) → integer, pence
- `fareCategory` (NS_COM) → `ADVANCE` | `OFF-PEAK` | `ANYTIME`
- `description` (NS_COM) → fallback fare label
- `departure` / `arrival` in `timetable/scheduled` → ISO datetime strings

### Fare Selection

Collects ALL fares across all returned journeys. Finds minimum price. Among fares at that minimum price, prefers the **first departure at or after 06:00** (avoids showing the 05:40 first train when all prices are identical).

### Date Parsing (`format_datetime`)

Uses `dateparser` with:
- `PREFER_DATES_FROM: future`
- `DATE_ORDER: DMY`
- `RETURN_AS_TIMEZONE_AWARE: False`

**Critical**: If input already matches `^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}`, skip re-parsing entirely — prevents day/month swap bug (ISO dates fed back in were being re-parsed with DMY order).

If user mentioned a time (`\d+:\d+` or `\d+(am|pm)`), preserve it. Otherwise override with `hour=9` default.

Fallback: strptime with ordinal stripping (`st`, `nd`, `rd`, `th` removed).

### Booking URL

```python
def _nr_booking_url(origin_crs, destination_crs, iso_datetime, ticket_type="single"):
    # National Rail journey planner
    # Uses 'from'/'to' params (not 'origin'/'destination') for stations
    return (
        f"https://www.nationalrail.co.uk/journey-planner/"
        f"?type={journey_type}"
        f"&from={origin_crs}&to={destination_crs}"
        f"&leavingType=departing"
        f"&leavingDate={date_str}"
        f"&leavingHour={hour_str}&leavingMin={min_str}"
        f"&adults=1&children=0&extraTime=0"
    )
```

**Graveyard of failed approaches**:
- `ojp.nationalrail.co.uk/service/timesandfares/` → not a web UI
- `nationalrail.co.uk/journey-planner/?origin=NRW` → `origin`/`destination` not recognised, showed " **to** " blank
- Trainline with CRS codes → not recognised
- Trainline with station names + `quote_plus` + `outwardDateType=departAfter` → blank SPA page

**Current approach**: NR journey planner with `from`/`to` + CRS codes. Date, time, adults load correctly; stations should now load with `from`/`to` (not yet confirmed working).

**Root cause**: Both NR and Trainline are React SPAs. They selectively read some URL params and ignore others depending on their internal router. No approach has been 100% reliable.

---

## Task 2: Delay Prediction (`task2/`)

- Route: Weymouth → London Waterloo (SWR)
- Model: Keras neural network trained on `train_performance` table
- Features: current station CRS, destination CRS, day of week, month, planned arrival
- Output: predicted delay in minutes + estimated arrival + confidence
- **Must retrain on demo machine** — model files are gitignored

---

## Database (`database.py`)

### Tables

- `conversation_history` — logs all exchanges
- `station_codes` — `(station_name, crs_code)` — seeded from CSV at startup
- `train_performance` — historical delay data for Task 2
- `user_sessions` — session tracking
- `kb_qa` — Knowledge Base Q&A pairs
- `kb_rules` — expert system rules (JSON)
- `kb_fallbacks` — fallback responses

### Station seeding

```python
# Runs once per session via st.session_state.db_ready guard
initialise_database()
seed_station_codes()  # loads ~2,600 stations from data/StationNameAndCode.csv
```

CSV format: `station_name,crs_code` (no header). Names `.title()`-cased, CRS `.upper()`-ed.

---

## Knowledge Acquisition Flow

Triggered by `INTENT_ADD_RULE`. Stages:

1. `ka_ask_category` → store category
2. `ka_ask_question` → store question/trigger phrase
3. `ka_ask_keywords` → comma-separated keywords
4. `ka_ask_answer` → store answer → call `kb.add_qa()` → confirm + reset

KB checked at `stage=None` before intent routing. If KB match found and message doesn't contain booking/delay trigger words, return KB answer directly.

---

## UI / CSS

Dark theme. Key custom CSS injected via `st.markdown`:

```css
[data-testid="stChatInput"] {
    background: #1a1a2e !important;
    border: 1.5px solid #2e3a5c !important;
}
[data-testid="stChatInput"] textarea {
    color: #f0f4f8 !important;
    background: #1a1a2e !important;
}
```

---

## Known Issues / Outstanding

### Booking URL (main unresolved issue)
After many attempts, NR journey planner with `from`/`to` CRS params is the current best guess. Has not been confirmed working end-to-end. Root problem: NR and Trainline are React SPAs that inconsistently read URL params.

### `chatbot.db` must be deleted on restart
Old seeded data (18 hardcoded stations) persists until DB is deleted. Always `rm chatbot.db` before testing after code changes.

### Git lock errors
Sandbox can't remove `.git/index.lock` or `.git/HEAD.lock`. User must run manually:
```bash
rm -f .git/index.lock .git/HEAD.lock
```

### Keras model not in repo
Must run on demo machine: `python task2/keras_model.py`

---

## Deadlines

| Item | Due |
|---|---|
| Demo (10–12 min presentation) | TBD |
| Group Report | 22 May 2026 |
| Individual Contribution Reports | 22 May 2026 |

---

## Environment

- `.env` file required: `NATIONAL_RAIL_USERNAME`, `NATIONAL_RAIL_PASSWORD`
- `python-dotenv` loads it in `ticket_api.py`
- Run: `streamlit run app.py`
- Packages: see `requirements.txt` (spacy, experta, scikit-learn, tensorflow, keras, pandas, numpy, streamlit, requests, joblib, python-dotenv, dateparser)
