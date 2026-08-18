# ✈️ Wanderlust Travels RAG Chatbot

AI-powered travel assistant using Azure OpenAI, Azure AI Search, and LLMOps best practices.

## 🌟 Features

*   **RAG-Powered Responses**: Retrieves information from travel policies, routes, and FAQs
*   **Azure AI Integration**: Uses Azure OpenAI (GPT-4) and Azure AI Search
*   **Ragas Evaluation**: Evaluates faithfulness, answer relevancy, context precision, and context recall
*   **Governance & Guardrails**:
    *   Safety Validator: Detects prompt injection and unsafe content
    *   Compliance Checker: Detects and redacts PII (GDPR compliance)
*   **Observability**: MLflow tracking and Azure Monitor integration
*   **Production-Ready**: Docker containerization for easy deployment

---

## 🛠️ Prerequisites

*   **Python 3.10+**
*   **Azure Subscription** with:
    *   Azure OpenAI Service (GPT-4 deployment)
    *   Azure AI Search (Free tier works)
    *   Azure Content Safety (Optional)
*   **OpenAI API Key** (for Ragas evaluation)

---

## 🚀 Setup & Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd wanderlust-travel-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```


### 4. Prepare Data
Place your travel documents **directly** in the `data/` folder (no subfolders):

```
data/
├── ai-schedule-change-policy-updated.pdf
├── air-india-coc.pdf
├── air-india-general-booking-policies-oct2025.pdf
├── U.S. Department of Transportation - Air Consumer Privacy.pdf
├── U.S. Department of Transportation - Aircraft Dissinection.pdf
├── U.S. Department of Transportation - Aviation Industry Bankruptcy and Service Cessation.pdf
├── U.S. Department of Transportation - Implementation of the Consumer Credit Protection Act With Respect to Air Carriers and Foreign Air Carriers.pdf
├── U.S. Department of Transportation - Refunds and Other Consumer Protections.pdf
├── U.S. Department of Transportation - Refunds for Airline Fare and Ancillary Service Fees.pdf
└── golden_dataset.json  (for evaluation)
```

---

## 🏃‍♂️ Usage

### 1. Start MLflow (Optional - for tracking)
```bash
./start_mlflow.sh
```
*   View dashboard at `http://localhost:5000`

**Troubleshooting MLflow:**
```bash
# If MLflow doesn't start, use restart script
./restart_mlflow.sh

# Or manually:
pkill -f "mlflow server"
mlflow server --host 127.0.0.1 --port 5000
```

### 2. Ingest Documents
Populate Azure AI Search index with your travel knowledge base:
```bash
python src/ingestion.py

or
python -m src.ingestion
```

**Output:**
```
📂 Loading travel knowledge base from data/ folder...
============================================================
📂 Found 9 PDF files in data/
  ✓ Loaded: ai-schedule-change-policy-updated.pdf (12 pages) - Category: air_india_policies
  ✓ Loaded: air-india-coc.pdf (8 pages) - Category: air_india_policies
  ...
============================================================
✅ Total documents loaded: 87

✂️  Splitting 87 documents into chunks...
✅ Created 245 chunks
   Average chunk size: 892 chars

📥 Indexing documents to Azure AI Search...
Indexing: 100%|████████████████| 5/5 [00:15<00:00,  3.2s/it]

✅ Ingestion Complete!
   Successfully indexed: 245 chunks
```


### 3. Run the Application
Launch the Streamlit interface:
```bash
streamlit run src/app.py
```
*   Access at `http://localhost:8501`

### 4. Test Queries
Try these example queries:
- "What are the baggage rules for international flights?"
- "Do I need a visa to travel from India to UK?"
- "What is the cancellation policy for Air India?"
- "What is the refund policy if my flight is delayed?"
- "What countries require aircraft disinsection?"
---


## 📊 Evaluation (Ragas)

### Run Evaluation
Evaluate chatbot performance using Ragas metrics:
```bash
python src/evaluate.py
```

**Output:**
```
======================================================================
Starting Ragas Evaluation...
======================================================================
Loaded 15 test cases

Generating responses...
DEBUG: Text Query: What is the specific reimbursement percentage for an Air India passenger who is involuntarily downgraded on a domestic flight?
...

Running Ragas metrics...
Metrics: faithfulness, answer_relevancy, context_precision, context_recall

======================================================================
EVALUATION RESULTS
======================================================================

Ragas Scores:
  Faithfulness:       0.8542
  Answer Relevancy:   0.8123
  Context Precision:  0.7891
  Context Recall:     0.8267
  Category Accuracy:  0.8667
======================================================================

✅ Evaluation summary saved to reports/evaluation_summary.json
✅ Detailed results saved to reports/evaluation_detailed.csv
✅ Category breakdown saved to reports/category_breakdown.json

✅ EVALUATION PASSED
```

### Evaluation Metrics Explained

The system uses **Ragas** to evaluate RAG performance:

1. **Faithfulness (0-1)**: Measures if the answer is grounded in retrieved context
   - Checks for hallucinations or unsupported claims
   - **Threshold**: ≥ 0.70

2. **Answer Relevancy (0-1)**: Measures if the answer addresses the question
   - Evaluates how well the response answers what was asked
   - **Threshold**: ≥ 0.70

3. **Context Precision (0-1)**: Measures quality of retrieved documents
   - Evaluates if the top-ranked documents are most relevant
   - Higher score = better ranking

4. **Context Recall (0-1)**: Measures if all relevant info was retrieved
   - Checks if retrieved context contains all information needed
   - Higher score = more complete retrieval

5. **Category Accuracy (0-1)**: Custom metric for document categorization
   - Measures if retrieved documents match expected category
   - Based on the `_categorize_document()` logic

### Golden Dataset Structure

Located at: `data/golden_dataset.json`

Format:
```json
{
  "question": "What is the refund policy?",
  "ground_truth": "Expected answer...",
  "contexts": ["Supporting context from documents..."],
  "source": "policy_document.pdf",
  "category": "refund_policies"
}
```

**Categories** (based on `data_loader.py`):
- `air_india_policies`: Air India specific policies
- `us_dot_regulations`: U.S. Department of Transportation regulations
- `booking_policies`: General booking and reservation policies
- `refund_policies`: Refund and compensation policies
- `privacy_policies`: Privacy and data protection policies
- `general`: General travel information

---


## 🔍 Testing

### Run All Tests
```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov=governance --cov=guardrails

# Run specific test file
pytest tests/test_search_engine.py -v

# Run with output
pytest tests/ -v -s
```

### Test Coverage
Current test coverage includes:
- Configuration loading
- Search engine initialization
- Document loading
- Governance checks
- Vector store integration

---


## 🛡️ Governance & Guardrails

The system implements multi-layer safety:

### 1. **Input Validation** (Before Processing)
   - **PII Detection**: Detects email, phone, passport, SSN, credit card
   - **Prompt Injection Detection**: Blocks malicious prompts
   - **Content Safety**: Azure Content Safety + keyword filtering
   - **Blocked queries** return: "Query blocked by security checks"

### 2. **Output Validation** (After Generation)
   - **PII Leakage Prevention**: Ensures no sensitive info in response
   - **Unsafe Content Filtering**: Blocks inappropriate content
   - **Blocked responses** return: "Response didn't pass safety checks"

### 3. **Compliance**
   - **GDPR Compliance**: PII detection and redaction
   - **Audit Logging**: All governance checks logged
   - **Transparency**: Clear violation messages
```

---
## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build image
docker build -t wanderlust-chatbot .

# Run container
docker run -p 8501:8501 \
  -e AZURE_OPENAI_API_KEY="your-key" \
  -e AZURE_SEARCH_ENDPOINT="your-endpoint" \
  -e AZURE_SEARCH_KEY="your-key" \
  wanderlust-chatbot
```

### Deploy to Cloud
The Dockerfile is ready for deployment to:
- Azure Container Instances
- Azure App Service
- Azure Kubernetes Service
- Any container platform

---

## 📂 Project Structure

```
wanderlust-travel-chatbot/
├── governance/              # Governance & compliance
│   ├── __init__.py
│   ├── compliance_checker.py   # PII detection and GDPR compliance
│   ├── governance_gate.py      # Main governance orchestrator
│   └── safety_validator.py     # Content safety and prompt injection detection
├── guardrails/              # Safety guardrails
│   ├── __init__.py
│   ├── content_safety.py   # Keyword-based content filtering
│   └── pii_detector.py     # Regex-based PII detection
├── src/                     # Application code
│   ├── monitoring/          # Logging & metrics
│   │   ├── __init__.py
│   │   ├── logger.py           # Structured JSON logging
│   │   ├── metrics_collector.py # MLflow metrics tracking
│   │   └── trace_collector.py  # OpenTelemetry tracing
│   ├── app.py              # Streamlit UI
│   ├── config.py           # Configuration management
│   ├── data_loader.py      # Document loading from data/
│   ├── ingestion.py        # Index creation and batch ingestion
│   ├── search_engine.py    # RAG search engine
│   ├── vector_store.py     # Azure AI Search wrapper
│   └── evaluate.py         # Ragas evaluation script
├── tests/                  # Unit tests
│   ├── __init__.py
│   ├── test_config.py
│   ├── test_search_engine.py
│   └── test_search.py
├── data/                   # Knowledge base (PDFs directly here)
│   ├── *.pdf               # All travel policy documents
│   └── golden_dataset.json # Evaluation test cases
├── reports/                # Generated evaluation reports
│   ├── evaluation_summary.json
│   ├── evaluation_detailed.csv
│   └── category_breakdown.json
├── .env                    # Environment variables (not in git)
├── env.sample              # Sample environment file
├── Dockerfile              # Container definition
├── requirements.txt        # Python dependencies
├── start_mlflow.sh         # MLflow startup script
├── restart_mlflow.sh       # MLflow restart script
└── README.md              # This file
```

---

## 🎯 Architecture
```
User Query
    ↓
Governance Gate (Input Validation)
    ↓
RAG Pipeline
    ├─ Azure AI Search (Vector Retrieval)
    └─ Azure OpenAI GPT-4 (Generation)
    ↓
Governance Gate (Output Validation)
    ↓
Response to User
```

**Components:**

1. **Data Loading**: `TravelDataLoader` loads PDFs from `data/` folder
2. **Document Categorization**: Auto-categorizes based on filename
3. **Text Chunking**: RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
4. **Embeddings**: Azure OpenAI text-embedding-3-small
5. **Vector Store**: Azure AI Search (no ChromaDB)
6. **LLM**: Azure OpenAI GPT-4
7. **Governance**: Multi-layer input/output validation
8. **Monitoring**: MLflow + Azure Monitor

---

## 📊 Monitoring & Observability

### MLflow Tracking
- **Experiments**: Track ingestion and search operations
- **Parameters**: Log query parameters, model configs
- **Metrics**: Track latency, token usage, retrieval count
- **Artifacts**: Save responses and evaluation results

**Access MLflow UI:**
```bash
./start_mlflow.sh
# Open http://localhost:5000
```

### Azure Monitor (Optional)
- **Application Insights**: Track requests, exceptions, dependencies
- **Structured Logging**: JSON logs with context
- **Distributed Tracing**: OpenTelemetry integration

### Logging
All components use structured JSON logging:
```python
from src.monitoring import logger

logger.info("Search completed", 
    query="baggage rules",
    results_count=5,
    latency_ms=245
)
```

---

## 🔧 Troubleshooting

### Issue: No documents indexed
```bash
# Verify data directory structure
ls -R data/

# Re-run ingestion
python src/ingestion.py
```

### Issue: Search returns no results
```bash
# Test Azure AI Search connection
python -c "from src.search_engine import TravelSearchEngine; engine = TravelSearchEngine(); print('Connected!')"
```

### Issue: MLflow not starting
```bash
# Kill existing processes
pkill -f mlflow

# Start fresh
./restart_mlflow.sh
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest tests/`
5. Submit a pull request

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments
- Azure AI Services
- LangChain
- Streamlit
- MLflow
```# AIWithMicrosoftAzure
# AIWithMicrosoftAzure
