"""
Knowledge Base Usage Examples

Demonstrates how the KB can be used in the chatbot workflow.
"""

from engine.knowledge_base import get_kb

########### EXAMPLE 1: Basic Q&A Lookup

def example_basic_lookup():
    print("=" * 70)
    print("EXAMPLE 1: Basic Q&A Lookup")
    print("=" * 70)

    kb = get_kb()

    user_queries = [
        "What ticket types are available?",
        "Can I cancel my booking?",
        "How do I claim delay compensation?",
        "How do I contact customer service?",
    ]

    for query in user_queries:
        print(f"\nUser: {query}")
        match = kb.search_qa(query)

        if match:
            print(f"KB Match: {match['question']}")
            print(f"Answer: {match['answer']}\n")
        else:
            print(f"No match found. Using fallback...")
            print(f"Fallback: {kb.get_fallback_response(0)}\n")

############# EXAMPLE 2: Using Business Rules (Retrieve and apply business rules.)

def example_business_rules():
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Business Rules Application")
    print("=" * 70)

    kb = get_kb()

    # Scenario: User wants to cancel an advance ticket
    print("\nScenario: User cancelling advance ticket")

    refund_rule = kb.get_rule("refund_eligibility", "advance_ticket")
    print(f"Rule: {refund_rule['description']}")
    print(f"Refundable: {refund_rule['refundable']}")
    print(f"Exchange available: {refund_rule['exchange']}")
    print(f"Exchange fee: {refund_rule['exchange_fee']}")

    bot_response = (
        "I'm sorry you want to cancel. Advance tickets are non-refundable, "
        f"but you can exchange it for a {refund_rule['exchange_fee']} fee. "
        "Would you like to exchange instead?"
    )
    print(f"\nBot: {bot_response}")

    # Scenario: User qualifies for delay compensation
    print("\n\nScenario: User train was 45 minutes late")

    delay_thresholds = kb.get_rule("delay_compensation_thresholds")
    delay_minutes = 45

    compensation_pct = None
    for threshold, details in delay_thresholds.items():
        if threshold == "30_59_mins":
            compensation_pct = details["percentage"]

    print(f"Delay: {delay_minutes} minutes")
    print(f"Compensation: {compensation_pct}% of ticket price")

    bot_response = (
        f"Your train was {delay_minutes} minutes late. "
        f"You're eligible for {compensation_pct}% Delay Repay compensation! "
        f"Would you like to file a claim?"
    )
    print(f"Bot: {bot_response}")

############### EXAMPLE 3: Dynamic KB Updates

def example_knowledge_acquisition():
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Knowledge Acquisition")
    print("=" * 70)

    kb = get_kb()

    # Scenario: Support team wants to add new FAQ
    print("\nScenario: Support team adds new FAQ about railcards")

    print("\n1) Adding new Q&A...")
    kb.add_qa(
        category="ticket_types",
        question="What discounts do railcards provide?",
        keywords=["railcard", "discount", "young person", "senior"],
        answer=(
            "Railcards provide 1/3 off most fares:\n"
            "- Young Person Railcard (16-30): 1/3 off\n"
            "- Senior Railcard (60+): 1/3 off\n"
            "- Family & Friends Railcard: 20% off for adults, 60% off children\n"
            "Annual cost: £30. Savings usually pay for itself within 3-4 journeys."
        ),
    )
    print("Added successfully")

    # Verify it was added
    print("\n2) Checking new Q&A...")
    match = kb.search_qa("railcard discount")
    if match:
        print(f"Found: {match['question']}")
        print(f"Answer preview: {match['answer'][:50]}...") # Showing only the first 50 characters of the answer...

    # Update KB statistics
    print("\n3) KB Statistics:")
    summary = kb.get_kb_summary()
    print(f"Total Q&As: {summary['total_qa_pairs']}")
    print(f"Categories: {summary['categories']}")
    print(f"Last updated: {summary['last_updated']}")

    # Clean up
    print("\n4) Cleaning up test Q&A...")
    kb.delete_qa("ticket_types", "What discounts do railcards provide?")
    print("Removed")

############## EXAMPLE 4: Dialogue Flow with KB

def example_dialogue_flow():
    """Example 4: Multi-turn dialogue using KB."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Multi-Turn Dialogue Flow")
    print("=" * 70)

    kb = get_kb()

    # Simulate an example multi-turn conversation
    dialogue = [
        ("User", "Hi, I'd like to book a train ticket"),
        ("Bot", "Sure! I can help you find the cheapest ticket. Where are you travelling from?"),
        ("User", "Norwich to London"),
        ("Bot", "Great! What date would you like to travel?"),
        ("User", "Next Friday"),
        ("Bot", "Would you like a single or return ticket?"),
        ("User", "Single, but what's the difference?"),
        ("KB_QUERY", "What is a single ticket?"),
    ]

    for speaker, message in dialogue:
        if speaker == "User":
            print(f"\nUser: {message}")
        elif speaker == "Bot":
            print(f"Bot: {message}")
        elif speaker == "KB_QUERY":
            print(f"\n[Internal: Querying KB for: '{message}']")
            match = kb.search_qa(message)
            if match:
                print(f"Bot: {match['answer']}\n")

########### EXAMPLE 5: Handling Different User Intents (Route user queries to appropriate KB categories)

def example_intent_routing():
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Intent-Based Query Routing")
    print("=" * 70)

    kb = get_kb()

    intent_queries = {
        "booking_intent": "I want to book a ticket",
        "refund_intent": "Can I get a refund?",
        "delay_intent": "My train is delayed",
        "service_intent": "What facilities do you have?",
        "contact_intent": "How do I contact support?",
    }

    for intent, query in intent_queries.items():
        print(f"\nIntent: {intent}")
        print(f"Query: '{query}'")

        match = kb.search_qa(query)
        if match:
            category = None
            for cat, qa_list in kb.qa_database.items():
                if match in qa_list:
                    category = cat
                    break

            print(f"Category: {category}")
            print(f"Answer: {match['answer'][:60]}...")
        else:
            print("No direct match, would trigger fallback")

########### EXAMPLE 6: KB Statistics and Reporting

def example_kb_analytics():
    """Example 6: Generate KB statistics and reports."""
    print("\n" + "=" * 70)
    print("EXAMPLE 6: KB Analytics and Reporting")
    print("=" * 70)

    kb = get_kb()

    summary = kb.get_kb_summary()

    print(f"\nKB Summary Report\n")
    print(f"Total Q&A pairs: {summary['total_qa_pairs']}")
    print(f"Categories: {summary['categories']}")
    print(f"Business rules: {summary['rules']}")
    print(f"Last updated: {summary['last_updated']}")

    print(f"\nCategory Breakdown:\n")
    for category, count in sorted(summary["category_breakdown"].items()):
        percentage = (count / summary["total_qa_pairs"]) * 100
        print(f"{category:.<30} {count:>2} Q&As {'.' * 15:.<15} {percentage:>5.1f}%")

########### EXAMPLE 7: Error Handling and Edge Cases

def example_edge_cases():
    """Example 7: Handle edge cases and errors."""
    print("\n" + "=" * 70)
    print("EXAMPLE 7: Edge Cases and Error Handling")
    print("=" * 70)

    kb = get_kb()

    test_cases = [
        ("", "Empty query"),
        ("xyz abc def", "No matching keywords"),
        ("ticket", "Partial/vague match"),
        ("What is a SINGLE TICKET?", "Case insensitivity"),
        ("   what   is   ticket   ", "Extra whitespace"),
    ]

    for query, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Query: '{query}'")

        match = kb.search_qa(query)
        if match:
            print(f"Match found: {match['question']}")
        else:
            print(f"No match, fallback: {kb.get_fallback_response(0)[:50]}...")


############# EXAMPLE 8: Integration with External Systems

def example_external_integration():
    """Example 8: Integrate KB with external data sources."""
    print("\n" + "=" * 70)
    print("EXAMPLE 8: External System Integration")
    print("=" * 70)

    kb = get_kb()

    # Get contact info from KB to use in error handling
    print("\nUse Case: Display support contact when chatbot can't help")
    contact_qa = kb.search_qa("contact SWR")
    if contact_qa:
        print(f"Support Information from KB:")
        print(f"{contact_qa['answer']}")

    # Get delay compensation info to inform customer
    print("\n\nUse Case: Show compensation info when delay is detected")
    delay_qa = kb.search_qa("Delay Repay")
    if delay_qa:
        print(f"Information shown to delayed customer:")
        print(f"{delay_qa['answer']}")

    # Get policies for booking confirmation
    print("\n\nUse Case: Show policies during booking confirmation")
    policy_qa = kb.search_qa("Can I change my ticket?")
    if policy_qa:
        print(f"Policy shown before purchase:")
        print(f"{policy_qa['answer']}")


################ MAIN

if __name__ == "__main__":
    print("\n" + "#" * 70)
    print("KB USAGE EXAMPLES")
    print("#" * 70 + "\n")

    examples = [
        example_basic_lookup,
        example_business_rules,
        example_knowledge_acquisition,
        example_dialogue_flow,
        example_intent_routing,
        example_kb_analytics,
        example_edge_cases,
        example_external_integration,
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Error in {example_func.__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "#" * 70)
    print("Examples completed")
    print("#" * 70 + "\n")
