# Knowledge Base System Documentation

## 1. Introduction

The Knowledge Base (KB) is a critical component of the AI chatbot system, responsible for storing and retrieving domain-specific information related to railway services. It addresses the limitations of rule-based systems by providing a flexible, extensible knowledge repository that supports:

- **Question-Answer Pairs**: Structured responses to frequently asked questions
- **Business Rules**: Encoded policies and decision-making logic
- **Fallback Mechanisms**: Graceful degradation for unrecognized queries
- **Knowledge Acquisition**: Dynamic updates to the knowledge base

## 2. System Architecture

### 2.1 Core Components

The KB system is implemented as a singleton class in `engine/knowledge_base.py`, ensuring consistent state across the application. The architecture comprises the following components:

```
KnowledgeBase Class
├── Q&A Database
│   ├── ticket_types (5 entries)
│   ├── booking_policies (3 entries)
│   ├── cancellation_policies (3 entries)
│   ├── delay_compensation (3 entries)
│   ├── service_info (3 entries)
│   └── toc_contact (2 entries)
├── Business Rules Engine
│   ├── refund_eligibility
│   ├── delay_compensation_thresholds
│   └── delay_exceptions
├── Fallback Response System (3 responses)
└── Knowledge Acquisition API
    ├── add_qa()
    ├── update_qa()
    ├── delete_qa()
    └── add_rule()
```

### 2.2 Data Structures

#### 2.2.1 Q&A Database Format

Each Q&A entry is structured as a dictionary with the following schema:

```python
{
    "question": str,        # The question text
    "keywords": List[str],  # Keywords for semantic matching
    "answer": str          # The response text
}
```

**Example:**
```python
{
    "question": "What is a single ticket?",
    "keywords": ["single ticket", "single fare", "one way"],
    "answer": "A single ticket covers one outbound journey only. You can travel any date/time within the validity period. Usually valid for 7 days after issue. Good for one-off trips."
}
```

#### 2.2.2 Business Rules Structure

Business rules are stored as nested dictionaries containing category-specific policy data:

```python
# Refund Eligibility Rules
{
    "advance_ticket": {
        "refundable": False,
        "exchange": True,
        "exchange_fee": "£10-30"
    },
    "single_ticket": {
        "refundable": True,
        "exchange": True,
        "exchange_fee": "£5"
    }
}

# Delay Compensation Thresholds
{
    "15_29_mins": {"percentage": 25, "description": "15-29 minute delay"},
    "30_59_mins": {"percentage": 50, "description": "30-59 minute delay"},
    "60_plus_mins": {"percentage": 100, "description": "60+ minute delay"}
}
```

## 3. Knowledge Organization

### 3.1 Q&A Categories

The knowledge base is organized into semantically coherent categories aligned with the chatbot's primary functions:

#### 3.1.1 Ticket Booking Domain
- **ticket_types**: Fundamental ticket classifications and definitions
- **booking_policies**: Reservation procedures and constraints
- **cancellation_policies**: Refund and cancellation procedures

#### 3.1.2 Service Operations Domain
- **delay_compensation**: Compensation schemes for service disruptions
- **service_info**: Operational information and facilities
- **toc_contact**: Customer service contact information

### 3.2 Category Contents

| Category | Entries | Sample Questions |
|----------|---------|------------------|
| ticket_types | 5 | What ticket types do you offer? What is a single ticket? |
| booking_policies | 3 | When can I book a ticket? Can I change my ticket? |
| cancellation_policies | 3 | Can I cancel my ticket? How do I get a refund? |
| delay_compensation | 3 | What is Delay Repay? How do I claim compensation? |
| service_info | 3 | What routes do you operate? What facilities are available? |
| toc_contact | 2 | How do I contact customer service? |

## 4. Implementation Details

### 4.1 Search Mechanism

The `search_qa()` method implements a multi-stage search algorithm:

1. **Keyword Matching** (Priority: High, Score: 0.9)
   - Performs substring matching against keyword lists
   - Most reliable for exact intent recognition

2. **Question Similarity** (Priority: Medium, Score: Variable)
   - Calculates word overlap between query and stored questions
   - Handles paraphrased queries

3. **Threshold Filtering** (Default: 0.6)
   - Ensures minimum confidence before returning matches
   - Configurable for different precision requirements

```python
# Example usage with different thresholds
strict_match = kb.search_qa("delayed train compensation", threshold=0.8)
loose_match = kb.search_qa("train delay", threshold=0.4)
```

### 4.2 Singleton Pattern Implementation

The system employs the singleton design pattern to maintain data consistency:

```python
class KnowledgeBase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# Global access function
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
```

This ensures that all components interact with the same KB instance, preventing data inconsistencies.

### 4.3 Database Integration

The KB integrates with SQLite for persistent storage:

- **Tables**: kb_qa, kb_rules, kb_fallbacks
- **Operations**: Automatic seeding on first run
- **Knowledge Acquisition**: All CRUD operations persist to database

## 5. Usage and Integration

### 5.1 Basic Operations

```python
from engine.knowledge_base import get_kb

kb = get_kb()

# Query resolution
result = kb.search_qa("What is a single ticket?")
if result:
    response = result["answer"]

# Rule retrieval
refund_policy = kb.get_rule("refund_eligibility", "advance_ticket")

# Fallback handling
fallback = kb.get_fallback_response(0)
```

### 5.2 Reasoning Engine Integration

Integration with the Experta reasoning engine enables rule-based query processing:

```python
from experta import KnowledgeEngine, Fact, Rule
from engine.knowledge_base import get_kb

class QueryEngine(KnowledgeEngine):
    @Rule(Fact(query=MATCH.question))
    def process_query(self, question):
        kb = get_kb()
        match = kb.search_qa(question)
        if match:
            self.response = match["answer"]
        else:
            self.response = kb.get_fallback_response(0)
```

### 5.3 Knowledge Acquisition API

Dynamic knowledge updates are supported through the acquisition API:

```python
# Add new Q&A pair
success = kb.add_qa(
    category="ticket_types",
    question="What is a group ticket?",
    keywords=["group", "bulk booking"],
    answer="Group tickets provide discounted fares for parties of 10+ passengers."
)

# Update existing entry
kb.update_qa(
    category="ticket_types",
    question="What is a single ticket?",
    new_answer="Updated definition with additional details..."
)

# Remove entry
kb.delete_qa("ticket_types", "What is a group ticket?")
```

## 6. Performance Analysis

### 6.1 Complexity Metrics

- **Search Time Complexity**: O(n) where n = total Q&A pairs
- **Keyword Matching**: O(m) where m = keywords per entry
- **Memory Footprint**: < 1MB (serialized JSON equivalent)
- **Database Operations**: O(log n) for indexed queries

### 6.2 Optimization Strategies

1. **Keyword Indexing**: Pre-computed keyword sets for fast lookup
2. **Caching**: Frequently accessed entries cached in memory
3. **Batch Operations**: Bulk updates for efficiency
4. **Threshold Tuning**: Configurable matching sensitivity

## 7. Testing and Validation

### 7.1 Test Suite

Comprehensive testing is implemented through:

```bash
# Unit tests for KB functionality
python engine/knowledge_base.py

# Integration examples
python kb_examples.py
```

### 7.2 Test Coverage

- **Basic Q&A Lookup**: 100% coverage across categories
- **Business Rules Application**: Policy enforcement validation
- **Knowledge Acquisition**: CRUD operation verification
- **Edge Cases**: Empty queries, no matches, malformed input
- **Performance**: Response time under load

## 8. Extension Mechanisms

### 8.1 Adding New Categories

```python
# Programmatic category addition
kb.qa_database["new_category"] = [
    {
        "question": "Sample question?",
        "keywords": ["keyword1", "keyword2"],
        "answer": "Sample answer"
    }
]
```

### 8.2 Rule Extensions

```python
# Adding new business rules
kb.add_rule("compensation_policy", {
    "express_trains": {
        "multiplier": 2.0,
        "description": "Enhanced compensation for express services"
    }
})
```

## 9. Best Practices

1. **Keyword Selection**: Use 2-4 semantically diverse keywords per entry
2. **Answer Conciseness**: Maintain responses under 200 words for readability
3. **Category Organization**: Limit to 5-10 categories for maintainability
4. **Regular Updates**: Implement version control for KB changes
5. **Testing**: Validate new entries against existing test cases
6. **Backup Strategy**: Regular JSON exports for disaster recovery

## 10. Future Enhancements

### 10.1 Planned Features

- **Automated Knowledge Acquisition**: Web crawling and API integration
- **Multi-language Support**: Localization capabilities
- **User Feedback Integration**: Rating and improvement systems
- **Advanced Search**: Semantic similarity using embeddings
- **Analytics Dashboard**: Usage statistics and performance metrics

### 10.2 Research Directions

- **Machine Learning Integration**: Neural retrieval models
- **Knowledge Graph**: Structured relationship modeling
- **Conversational Memory**: Context-aware response generation

## References

1. Russell, S., & Norvig, P. (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
2. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.). Prentice Hall.
3. Experta Documentation. https://experta.readthedocs.io/

---

*This documentation was prepared as part of the CMP6059B AI Chatbot project. Last updated: May 2026*
