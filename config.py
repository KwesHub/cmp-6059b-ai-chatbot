import os

# ─── Database ───────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "chatbot.db")

# ─── NLP ────────────────────────────────────────────────
SPACY_MODEL = "en_core_web_sm"

# ─── Intents ────────────────────────────────────────────
INTENT_BOOK_TICKET = "book_ticket"
INTENT_PREDICT_DELAY = "predict_delay"
INTENT_UNKNOWN = "unknown"

# ─── Task 2 route ───────────────────────────────────────
WEYMOUTH_TO_WATERLOO_STOPS = [
    "Weymouth",
    "Dorchester South",
    "Wool",
    "Wareham",
    "Poole",
    "Bournemouth",
    "Southampton Central",
    "Winchester",
    "London Waterloo"
]

# ─── Model paths ────────────────────────────────────────
BEST_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "best_model.pkl"
)
KERAS_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "keras_model.keras"
)

# ─── Data paths ─────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CLEANED_DATA_PATH = os.path.join(DATA_DIR, "cleaned_data.csv")