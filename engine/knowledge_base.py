from datetime import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from database import get_connection


class KnowledgeBase:
    def __init__(self):
        # in-memory store: {category: [{"question":..., "keywords":..., "answer":...}]}
        self.qa_database = {}
        self.rules = {}
        self.fallback_responses = []
        self.last_updated = datetime.now()

        # load Q&A pairs, business rules and fallback responses from SQLite on startup
        self._init_qa_database()
        self._init_rules()
        self._init_fallback_responses()

    # Q&A initialisation

    def _init_qa_database(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT category, question, keywords, answer FROM kb_qa")
        rows = cursor.fetchall()
        conn.close()

        # if the table is empty on first run, seed it with the defaults and reload
        if not rows:
            self._seed_qa_into_db()
            self._init_qa_database()
            return

        for category, question, keywords_json, answer in rows:
            if category not in self.qa_database:
                self.qa_database[category] = []
            keywords = json.loads(keywords_json) if keywords_json else []
            self.qa_database[category].append({
                "question": question,
                "keywords": keywords,
                "answer": answer
            })

    def _seed_qa_into_db(self):
        # writes the hard-coded default Q&As into the SQLite table once
        seed = self._get_seed_qa()
        conn = get_connection()
        cursor = conn.cursor()
        for category, qa_list in seed.items():
            for qa in qa_list:
                cursor.execute(
                    "INSERT INTO kb_qa (category, question, keywords, answer) VALUES (?, ?, ?, ?)",
                    (category, qa["question"], json.dumps(qa["keywords"]), qa["answer"])
                )
        conn.commit()
        conn.close()

    def _get_seed_qa(self):
        # default Q&A pairs that are loaded when the database is first created
        return {
            "ticket_types": [
                {
                    "question": "What ticket types do you offer?",
                    "keywords": ["ticket types", "what tickets", "which tickets"],
                    "answer": (
                        "We offer several ticket types:\n"
                        "- Single tickets - one journey, any date/time\n"
                        "- Return tickets - outbound + return, up to 30 days\n"
                        "- Advance tickets - cheapest, must book ahead, limited changes\n"
                        "- Off-peak tickets - cheaper fares during quiet times\n\n"
                        "Which type interests you?"
                    ),
                },
                {
                    "question": "What is a single ticket?",
                    "keywords": ["single ticket", "single fare", "one way"],
                    "answer": (
                        "A single ticket covers one outbound journey only. "
                        "You can travel any date/time within the validity period. "
                        "Usually valid for 7 days after issue. Good for one-off trips."
                    ),
                },
                {
                    "question": "What is a return ticket?",
                    "keywords": ["return ticket", "return fare", "round trip"],
                    "answer": (
                        "A return ticket includes your outbound journey and a return journey "
                        "within up to 30 days. Return journeys must be on a specified date or "
                        "within a date range. Often cheaper than two singles combined."
                    ),
                },
                {
                    "question": "What is an advance ticket?",
                    "keywords": ["advance ticket", "advance booking", "book ahead"],
                    "answer": (
                        "Advance tickets are discounted fares available for trains departing "
                        "at specific times. Must be purchased several days in advance. "
                        "Changes/cancellations may incur fees. Great savings for flexible travellers."
                    ),
                },
                {
                    "question": "What are off-peak tickets?",
                    "keywords": ["off-peak", "quiet time", "shoulder", "cheap fares"],
                    "answer": (
                        "Off-peak tickets are cheaper and valid during quieter travel times, "
                        "typically mid-morning to afternoon on weekdays, and off-peak hours weekends. "
                        "Exact times vary by route. Best for time-flexible travellers."
                    ),
                },
            ],
            "booking_policies": [
                {
                    "question": "When can I book a ticket?",
                    "keywords": ["book tickets", "when book", "booking time"],
                    "answer": (
                        "Tickets typically open for booking 12 weeks (84 days) in advance. "
                        "Advance and off-peak fares sell out quickly. "
                        "You can book online 24/7, up to departure time for some fares."
                    ),
                },
                {
                    "question": "Can I change my ticket?",
                    "keywords": ["change ticket", "modify booking", "alter ticket"],
                    "answer": (
                        "Changes are subject to your ticket type:\n"
                        "- Single/Return - often changeable with a fee\n"
                        "- Advance - limited changes, subject to availability\n"
                        "- Off-peak - may be changeable depending on specific restrictions\n\n"
                        "Check your ticket terms or contact us for details."
                    ),
                },
                {
                    "question": "What is the booking deadline?",
                    "keywords": ["book before", "deadline", "last booking time"],
                    "answer": (
                        "Most tickets can be purchased up to departure time online. "
                        "Station ticket offices close 1 hour before last train. "
                        "For travel on the same day, purchase at the station or before midnight online."
                    ),
                },
            ],
            "cancellation_policies": [
                {
                    "question": "Can I cancel my ticket?",
                    "keywords": ["cancel ticket", "cancellation", "refund ticket"],
                    "answer": (
                        "Cancellation depends on ticket type:\n"
                        "- Advance - Non-refundable, exchange only\n"
                        "- Off-peak - Non-refundable, exchange for a fee\n"
                        "- Single/Return - Often refundable up to departure minus a small fee\n\n"
                        "Check your ticket terms. Services disrupted due to delay may offer refunds."
                    ),
                },
                {
                    "question": "How do I get a refund?",
                    "keywords": ["refund", "money back", "compensation"],
                    "answer": (
                        "You can claim refunds through our online portal or by post:\n"
                        "1. Visit our website and enter your ticket reference\n"
                        "2. Select 'Claim Refund' and submit cancellation reason\n"
                        "3. Allow 10-14 business days for processing\n\n"
                        "Delays of 15+ mins may also qualify for Delay Repay compensation."
                    ),
                },
                {
                    "question": "What is your refund timeline?",
                    "keywords": ["refund time", "how long refund", "processing time"],
                    "answer": (
                        "Refunds typically take 10-14 business days after approval. "
                        "Check your bank or PayPal account. "
                        "Contact support if not received after 14 days."
                    ),
                },
            ],
            "delay_compensation": [
                {
                    "question": "What is Delay Repay?",
                    "keywords": ["delay repay", "compensation", "late train"],
                    "answer": (
                        "Delay Repay is our automatic compensation scheme:\n"
                        "- 15-29 minutes late -> 25% of ticket price refund\n"
                        "- 30-59 minutes late -> 50% of ticket price refund\n"
                        "- 60+ minutes late -> 100% of ticket price refund\n\n"
                        "File claims online within 28 days of travel."
                    ),
                },
                {
                    "question": "How do I claim delay compensation?",
                    "keywords": ["claim delay", "compensation process", "delay claim"],
                    "answer": (
                        "1. Visit www.swrailway.com/delays\n"
                        "2. Enter your ticket reference & journey date\n"
                        "3. Select compensation amount offered\n"
                        "4. Confirm payment method\n"
                        "5. Receive refund in 5-7 working days\n\n"
                        "Keep your ticket for proof. Claims accepted up to 28 days after travel."
                    ),
                },
                {
                    "question": "What delays qualify for compensation?",
                    "keywords": ["delay claim", "which delays", "qualify for"],
                    "answer": (
                        "Compensation is payable if:\n"
                        "- Train arrived 15+ minutes late at destination\n"
                        "- You hold a valid ticket for that journey\n"
                        "- Delay wasn't caused by: extreme weather, security threat, "
                        "passenger incidents, industrial action, or vandalism\n\n"
                        "Strikes and severe weather are exceptions."
                    ),
                },
            ],
            "service_info": [
                {
                    "question": "What routes do you operate?",
                    "keywords": ["routes", "where do you go", "services"],
                    "answer": (
                        "South Western Railway (SWR) operates trains across:\n"
                        "- South West - Weymouth, Dorchester, Bournemouth\n"
                        "- South Central - Southampton, Winchester\n"
                        "- London connections - Waterloo, Victoria, Clapham Junction\n\n"
                        "Check swrailway.com for full route map and timetables."
                    ),
                },
                {
                    "question": "How do I report a problem with my journey?",
                    "keywords": ["report issue", "complaint", "problem"],
                    "answer": (
                        "Report issues through:\n"
                        "- Online: www.swrailway.com/contact\n"
                        "- Phone: 0345 600 0650 (Mon-Fri 08:00-18:00)\n"
                        "- In person: Station ticket office\n"
                        "- Email: customer.services@swrailway.com\n\n"
                        "Provide ticket reference, date, time, and journey details."
                    ),
                },
                {
                    "question": "What on-board facilities do you offer?",
                    "keywords": ["facilities", "wifi", "toilet", "food", "onboard"],
                    "answer": (
                        "Most trains include:\n"
                        "- Standard & First Class seating\n"
                        "- Toilets & baby change facilities\n"
                        "- Food & beverage service\n"
                        "- Power sockets in First Class\n"
                        "- Bicycle racks (subject to space)\n\n"
                        "WiFi available on selected routes."
                    ),
                },
            ],
            "toc_contact": [
                {
                    "question": "How do I contact SWR?",
                    "keywords": ["contact", "phone number", "email", "reach us"],
                    "answer": (
                        "South Western Railway contact details:\n"
                        "Phone: 0345 600 0650 (Mon-Fri 08:00-18:00)\n"
                        "Email: customer.services@swrailway.com\n"
                        "Website: www.swrailway.com\n"
                        "Twitter: @SouthWesternRly\n\n"
                        "Report accessibility issues or need assistance? Call ahead."
                    ),
                },
                {
                    "question": "What are your customer service hours?",
                    "keywords": ["customer service", "hours", "open", "support"],
                    "answer": (
                        "Customer Services:\n"
                        "- Monday-Friday: 08:00-18:00 (phone)\n"
                        "- Saturday: 09:00-17:00\n"
                        "- Sunday: 10:00-16:00\n\n"
                        "Online services (website, mobile app) available 24/7."
                    ),
                },
            ],
        }

    # rules initialisation

    def _init_rules(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT rule_name, rule_data FROM kb_rules")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self._seed_rules_into_db()
            self._init_rules()
            return

        for rule_name, rule_data_json in rows:
            self.rules[rule_name] = json.loads(rule_data_json)

    def _seed_rules_into_db(self):
        seed = self._get_seed_rules()
        conn = get_connection()
        cursor = conn.cursor()
        for rule_name, rule_data in seed.items():
            cursor.execute(
                "INSERT INTO kb_rules (rule_name, rule_data) VALUES (?, ?)",
                (rule_name, json.dumps(rule_data))
            )
        conn.commit()
        conn.close()

    def _get_seed_rules(self):
        # business rules covering refund eligibility, delay thresholds and exceptions
        return {
            "refund_eligibility": {
                "advance_ticket": {
                    "refundable": False,
                    "exchange": True,
                    "description": "Advance tickets are non-refundable but can be exchanged for a fee.",
                },
                "off_peak_ticket": {
                    "refundable": False,
                    "exchange": True,
                    "description": "Off-peak tickets are non-refundable but can be exchanged.",
                },
                "single_return_ticket": {
                    "refundable": True,
                    "exchange": True,
                    "description": "Single/Return tickets are refundable up to departure.",
                },
            },
            "delay_compensation_thresholds": {
                "15_29_mins": {"percentage": 25, "description": "15-29 minute delay"},
                "30_59_mins": {"percentage": 50, "description": "30-59 minute delay"},
                "60_plus_mins": {"percentage": 100, "description": "60+ minute delay"},
            },
            "delay_exceptions": [
                "extreme weather",
                "security threat",
                "passenger incident",
                "industrial action",
                "vandalism",
            ],
            "booking_window_days": 84,
            "delay_claim_deadline_days": 28,
            "refund_processing_days": "10-14 business days",
        }

    # fallback initialisation

    def _init_fallback_responses(self):
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM kb_fallbacks ORDER BY id")
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            self._seed_fallbacks_into_db()
            self._init_fallback_responses()
            return

        self.fallback_responses = [row[0] for row in rows]

    def _seed_fallbacks_into_db(self):
        seed = self._get_seed_fallbacks()
        conn = get_connection()
        cursor = conn.cursor()
        for response in seed:
            cursor.execute("INSERT INTO kb_fallbacks (response) VALUES (?)", (response,))
        conn.commit()
        conn.close()

    def _get_seed_fallbacks(self):
        return [
            (
                "I'm not sure I understand that. I can help with:\n"
                "- Finding cheap train tickets (single, return, advance, off-peak)\n"
                "- Booking policies and cancellations\n"
                "- Delay compensation\n"
                "- Our service information and contact details\n\n"
                "What would you like to know?"
            ),
            (
                "That's outside my knowledge base right now. "
                "For general enquiries, please contact:\n"
                "12345 678910 (Mon-Fri 09:00-17:00)\n"
                "or email customerservice@email.com"
            ),
            (
                "I don't have specific information about that. "
                "Could you rephrase your question? I can help with tickets, bookings, "
                "delays, and service information."
            ),
        ]

    # retrieval

    def search_qa(self, query, threshold=0.6):
        """Search stored Q&A pairs by keyword match, then by word overlap as fallback."""
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for category, qa_pairs in self.qa_database.items():
            for qa in qa_pairs:
                score = 0

                # keyword match: if any stored keyword appears in the query, high confidence
                for keyword in qa.get("keywords", []):
                    if keyword.lower() in query_lower:
                        score = max(score, 0.9)
                        break

                # word overlap fallback: count how many question words appear in the query
                # and normalise by question length to get a 0-1 score
                if score == 0:
                    question_words = set(qa["question"].lower().split())
                    query_words = set(query_lower.split())
                    overlap = len(question_words & query_words)
                    if overlap > 0:
                        score = overlap / len(question_words)

                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = qa

        return best_match

    def get_rule(self, rule_name, rule_key=None):
        """Look up a business rule by name, optionally drilling into a sub-key."""
        rule = self.rules.get(rule_name)
        if rule_key and isinstance(rule, dict):
            return rule.get(rule_key)
        return rule

    def get_fallback_response(self, index=0):
        """Return a fallback response when the KB can't answer a query."""
        if 0 <= index < len(self.fallback_responses):
            return self.fallback_responses[index]
        return self.fallback_responses[0]

    # knowledge acquisition - called when the user says "add a rule"

    def add_qa(self, category, question, keywords, answer):
        """Add a new Q&A pair to the in-memory store and persist it to the DB."""
        if category not in self.qa_database:
            self.qa_database[category] = []

        self.qa_database[category].append({
            "question": question,
            "keywords": keywords,
            "answer": answer,
        })

        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO kb_qa (category, question, keywords, answer) VALUES (?, ?, ?, ?)",
            (category, question, json.dumps(keywords), answer)
        )
        conn.commit()
        conn.close()
        self.last_updated = datetime.now()
        return True


# single shared instance so the DB is only queried once per run
_kb_instance = None


def get_kb():
    """Return the shared KnowledgeBase instance, creating it on first call."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance
