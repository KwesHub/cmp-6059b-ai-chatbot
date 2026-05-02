from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import os


class KnowledgeBase:
    def __init__(self):
        """Initilise KB with seed data."""
        self.qa_database: Dict[str, List[Dict[str, str]]] = {}
        self.rules: Dict[str, Dict] = {}
        self.fallback_responses: List[str] = []
        self.last_updated = datetime.now()

        # Load initial KB
        self._init_qa_database()
        self._init_rules()
        self._init_fallback_responses()

    ################ Q&A DATABASE INTILISATION

    def _init_qa_database(self):
        """Initilise Q&A pairs organised by topic."""

        ### TASK 1: TICKET BOOKING

        self.qa_database["ticket_types"] = [
            {
                "question": "What ticket types do you offer?",
                "keywords": ["ticket types", "what tickets", "which tickets"],
                "answer": (
                    "We offer several ticket types:\n"
                    "• Single tickets — one journey, any date/time\n"
                    "• Return tickets — outbound + return, up to 30 days\n"
                    "• Advance tickets — cheapest, must book ahead, limited changes\n"
                    "• Off-peak tickets — cheaper fares during quiet times\n\n"
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
        ]

        self.qa_database["booking_policies"] = [
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
                    "• Single/Return — often changeable with a fee (£10-20)\n"
                    "• Advance — limited changes, £20-30 fee, subject to availability\n"
                    "• Off-peak — may be changeable depending on specific restrictions\n\n"
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
        ]

        self.qa_database["cancellation_policies"] = [
            {
                "question": "Can I cancel my ticket?",
                "keywords": ["cancel ticket", "cancellation", "refund ticket"],
                "answer": (
                    "Cancellation depends on ticket type:\n"
                    "• Advance — Non-refundable, exchange only for £10-30 fee\n"
                    "• Off-peak — Non-refundable, exchange for £15 fee\n"
                    "• Single/Return — Often refundable up to departure minus £5 fee\n\n"
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
        ]

        ### TASK 2: DELAYS & SERVICE INFO

        self.qa_database["delay_compensation"] = [
            {
                "question": "What is Delay Repay?",
                "keywords": ["delay repay", "compensation", "late train"],
                "answer": (
                    "Delay Repay is our automatic compensation scheme:\n"
                    "• 15-29 minutes late → 25% of ticket price refund\n"
                    "• 30-59 minutes late → 50% of ticket price refund\n"
                    "• 60+ minutes late → 100% of ticket price refund\n\n"
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
                    "• Train arrived 15+ minutes late at destination\n"
                    "• You hold a valid ticket for that journey\n"
                    "• Delay wasn't caused by: extreme weather, security threat, "
                    "passenger incidents, industrial action, or vandalism\n\n"
                    "Strikes and severe weather are exceptions."
                ),
            },
        ]

        self.qa_database["service_info"] = [
            {
                "question": "What routes do you operate?",
                "keywords": ["routes", "where do you go", "services"],
                "answer": (
                    "South Western Railway (SWR) operates trains across:\n"
                    "• South West — Weymouth, Dorchester, Bournemouth\n"
                    "• South Central — Southampton, Winchester\n"
                    "• London connections — Waterloo, Victoria, Clapham Junction\n\n"
                    "Check swrailway.com for full route map and timetables."
                ),
            },
            {
                "question": "How do I report a problem with my journey?",
                "keywords": ["report issue", "complaint", "problem"],
                "answer": (
                    "Report issues through:\n"
                    "• Online: www.swrailway.com/contact\n"
                    "• Phone: 0345 600 0650 (Mon-Fri 08:00-18:00)\n"
                    "• In person: Station ticket office\n"
                    "• Email: customer.services@swrailway.com\n\n"
                    "Provide ticket reference, date, time, and journey details."
                ),
            },
            {
                "question": "What on-board facilities do you offer?",
                "keywords": ["facilities", "wifi", "toilet", "food", "onboard"],
                "answer": (
                    "Most trains include:\n"
                    "• Standard & First Class seating\n"
                    "• Toilets & baby change facilities\n"
                    "• Food & beverage service\n"
                    "• Power sockets in First Class\n"
                    "• Bicycle racks (subject to space)\n\n"
                    "WiFi available on selected routes."
                ),
            },
        ]

        self.qa_database["toc_contact"] = [
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
                    "• Monday-Friday: 08:00-18:00 (phone)\n"
                    "• Saturday: 09:00-17:00\n"
                    "• Sunday: 10:00-16:00\n\n"
                    "Online services (website, mobile app) available 24/7."
                ),
            },
        ]

    def _init_rules(self):
        """Initilise business rules."""
        self.rules = {
            "refund_eligibility": {
                "advance_ticket": {
                    "refundable": False,
                    "exchange": True,
                    "exchange_fee": "£10-30",
                    "description": "Advance tickets are non-refundable but can be exchanged for a fee.",
                },
                "off_peak_ticket": {
                    "refundable": False,
                    "exchange": True,
                    "exchange_fee": "£15",
                    "description": "Off-peak tickets are non-refundable but can be exchanged.",
                },
                "single_return_ticket": {
                    "refundable": True,
                    "exchange": True,
                    "exchange_fee": "£5",
                    "description": "Single/Return tickets are refundable up to departure.",
                },
            },
            "delay_compensation_thresholds": {
                "15_29_mins": {
                    "percentage": 25,
                    "description": "15-29 minute delay",
                },
                "30_59_mins": {
                    "percentage": 50,
                    "description": "30-59 minute delay",
                },
                "60_plus_mins": {
                    "percentage": 100,
                    "description": "60+ minute delay",
                },
            },
            "delay_exceptions": [
                "extreme weather",
                "security threat",
                "passenger incident",
                "industrial action",
                "vandalism",
            ],
            "booking_window_days": 84,  # 12 weeks
            "delay_claim_deadline_days": 28,
            "refund_processing_days": "10-14 business days",
        }

    def _init_fallback_responses(self):
        """Initilise fallback responses for unrecognised queries."""
        self.fallback_responses = [
            (
                "I'm not sure I understand that. I can help with:\n"
                "• Finding cheap train tickets (single, return, advance, off-peak)\n"
                "• Booking policies and cancellations\n"
                "• Delay compensation\n"
                "• Our service information and contact details\n\n"
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

    ############## KNOWLEDGE RETRIEVAL

    def search_qa(self, query: str, threshold: float = 0.6) -> Optional[Dict]:
        """
        Search Q&A database using keyword matching.

        Args:
            query: User's question
            threshold: Minimum match score (0-1)

        Returns:
            Matching Q&A entry or None
        """
        query_lower = query.lower()
        best_match = None
        best_score = 0

        for category, qa_pairs in self.qa_database.items():
            for qa in qa_pairs:
                # Check keywords
                score = 0
                for keyword in qa.get("keywords", []):
                    if keyword.lower() in query_lower:
                        score = max(score, 0.9)  # High score for keyword match
                        break

                # Fallback: check question similarity
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

    def get_rule(self, rule_name: str, rule_key: Optional[str] = None):
        """
        Retrieve a business rule.

        Args:
            rule_name: Name of the rule (e.g. 'refund_eligibility')
            rule_key: Specific key within the rule (e.g. 'advance_ticket')

        Returns:
            Rule data or None
        """
        rule = self.rules.get(rule_name)
        if rule_key and isinstance(rule, dict):
            return rule.get(rule_key)
        return rule

    def get_fallback_response(self, index: int = 0) -> str:
        """Get a fallback response for unrecognised queries."""
        if 0 <= index < len(self.fallback_responses):
            return self.fallback_responses[index]
        return self.fallback_responses[0]

    ####### KNOWLEDGE ACQUISITION (Dynamic Updates)

    def add_qa(self, category: str, question: str, keywords: List[str], answer: str) -> bool:
        """
        Add a new Q&A pair to the KB.

        Args:
            category: Category name (e.g., 'ticket_types')
            question: The question
            keywords: List of keywords for matching
            answer: The answer

        Returns:
            True if successful
        """
        if category not in self.qa_database:
            self.qa_database[category] = []

        self.qa_database[category].append(
            {
                "question": question,
                "keywords": keywords,
                "answer": answer,
            }
        )
        self.last_updated = datetime.now()
        return True

    def update_qa(self, category: str, question: str, new_answer: str) -> bool:
        """
        Update an existing Q&A pair.

        Args:
            category: Category name
            question: The question to find
            new_answer: Updated answer

        Returns:
            True if successful
        """
        if category not in self.qa_database:
            return False

        for qa in self.qa_database[category]:
            if qa["question"].lower() == question.lower():
                qa["answer"] = new_answer
                self.last_updated = datetime.now()
                return True

        return False

    def add_rule(self, rule_name: str, rule_data: Dict) -> bool:
        """
        Add or update a business rule.

        Args:
            rule_name: Name of the rule
            rule_data: Rule data dictionary

        Returns:
            True if successful
        """
        self.rules[rule_name] = rule_data
        self.last_updated = datetime.now()
        return True

    def delete_qa(self, category: str, question: str) -> bool:
        """
        Delete a Q&A pair from the KB.

        Args:
            category: Category name
            question: The question to remove

        Returns:
            True if successful
        """
        if category not in self.qa_database:
            return False

        original_len = len(self.qa_database[category])
        self.qa_database[category] = [
            qa for qa in self.qa_database[category]
            if qa["question"].lower() != question.lower()
        ]

        if len(self.qa_database[category]) < original_len:
            self.last_updated = datetime.now()
            return True

        return False

    ####### PERSISTENCE (Save/Load)

    def save_to_file(self, filepath: str) -> bool:
        """Save KB to JSON file."""
        try:
            data = {
                "qa_database": self.qa_database,
                "rules": self.rules,
                "last_updated": self.last_updated.isoformat(),
            }
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving KB: {e}")
            return False

    def load_from_file(self, filepath: str) -> bool:
        """Load KB from JSON file."""
        try:
            if not os.path.exists(filepath):
                return False
            with open(filepath, "r") as f:
                data = json.load(f)
            self.qa_database = data.get("qa_database", {})
            self.rules = data.get("rules", {})
            return True
        except Exception as e:
            print(f"Error loading KB: {e}")
            return False

    ####### HELPERS

    def get_all_categories(self) -> List[str]:
        """Get all Q&A categories."""
        return list(self.qa_database.keys())

    def get_category_qa_count(self, category: str) -> int:
        """Get number of Q&A pairs in a category."""
        return len(self.qa_database.get(category, []))

    def get_kb_summary(self) -> Dict:
        """Get KB statistics."""
        total_qa = sum(len(qa_list) for qa_list in self.qa_database.values())
        return {
            "total_qa_pairs": total_qa,
            "categories": len(self.qa_database),
            "rules": len(self.rules),
            "last_updated": self.last_updated.isoformat(),
            "category_breakdown": {
                cat: len(qa_list)
                for cat, qa_list in self.qa_database.items()
            },
        }

################# GLOBAL INSTANCE

_kb_instance: Optional[KnowledgeBase] = None


def get_kb() -> KnowledgeBase:
    """Get or create the global KB instance (singleton pattern)."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
    return _kb_instance


########## MANUAL TEST

if __name__ == "__main__":
    kb = get_kb()

    print("KNOWLEDGE BASE TEST\n")

    print("Test 1: KB Summary")
    print(json.dumps(kb.get_kb_summary(), indent=2))

    print("\nTest 2: Search Q&A: 'What is a single ticket?'")
    match = kb.search_qa("What is a single ticket?")
    if match:
        print(f"Q: {match['question']}")
        print(f"A: {match['answer']}\n")

    print("Test 3: Search Q&A: 'delay compensation'")
    match = kb.search_qa("How do I claim delay compensation?")
    if match:
        print(f"Q: {match['question']}")
        print(f"A: {match['answer']}\n")

    print("Test 4: Get Business Rule: Delay compensation thresholds")
    rule = kb.get_rule("delay_compensation_thresholds")
    print(json.dumps(rule, indent=2))

    print("\nTest 5: Get Fallback Response")
    print(kb.get_fallback_response(0))

    print("\nTest 6: Add New Q&A")
    success = kb.add_qa(
        category="ticket_types",
        question="What is a group ticket?",
        keywords=["group ticket", "group fare", "group booking"],
        answer="Group tickets are available for groups of 10+ passengers, "
               "offering discounts of 20-30% off standard fares.",
    )
    print(f"Added: {success}")

    print("\nTest 7: Updated KB Summary")
    print(json.dumps(kb.get_kb_summary(), indent=2))

    print("\nTest 8: Search for newly added Q&A: 'group tickets'")
    match = kb.search_qa("group tickets")
    if match:
        print(f"Q: {match['question']}")
        print(f"A: {match['answer']}\n")