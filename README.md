# RAG System Development and Evaluation Project

A comprehensive Retrieval-Augmented Generation (RAG) system implementation with advanced features, rigorous evaluation, and detailed analysis. This project demonstrates the complete lifecycle of RAG system development from naive implementation to enhanced features and comprehensive evaluation using RAGAS metrics.

## 📋 Project Overview

This project implements and evaluates a RAG system with the following key components:

- **Naive RAG System**: Basic retrieval-augmented generation pipeline
- **Advanced Features**: Query rewriting and document reranking
- **Comprehensive Evaluation**: RAGAS metrics and parameter optimization
- **Detailed Analysis**: Performance comparison and insights

## 🏗️ Project Structure

```
├── data/                           # Dataset files
│   ├── training.csv               # Training passages (3,200 samples)
│   └── test.csv                   # Test questions (918 samples)
├── src/                           # Source code
│   ├── langchain_native_pipeline.py    # Core RAG implementation
│   ├── enhanced_rag_features.py       # Advanced RAG features
│   ├── RAGASCalculator.py             # RAGAS evaluation metrics
│   └── ragas_evaluation_clean.py       # Evaluation pipeline
├── deliverables/                  # Phase-wise deliverables
│   ├── phase1_domain documents/   # Dataset setup and analysis
│   ├── phase2_Naive RAG Implementation/  # Basic RAG system
│   ├── phase3/                    # Prompt strategy evaluation
│   ├── phase4/                    # Parameter optimization
│   ├── phase5-6/                  # Enhanced features and RAGAS evaluation
│   └── phase7/                    # Final summary report
├── notebooks/                     # Jupyter notebooks
│   └── deliverable1_dataset_setup.ipynb
├── requirements.txt              # Python dependencies
└── README.md                      # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key
- Google Generative AI API key (optional)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd assignment-2-copy
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_openai_api_key" > .env
echo "GOOGLE_API_KEY=your_google_api_key" >> .env  # Optional
```

### Basic Usage

1. **Run naive RAG system**
```python
from src.langchain_native_pipeline import build_native_rag_pipeline

# Build and test basic RAG system
rag_system = build_native_rag_pipeline(
    embedding_model="BAAI/bge-small-en-v1.5",
    training_csv="data/training.csv",
    output_dir="data/native_rag",
    llm_model="gpt-4o-mini"
)

# Query the system
answer = rag_system.query("What is the capital of Uruguay?")
print(answer)
```

2. **Run enhanced RAG system**
```python
from src.enhanced_rag_features import build_enhanced_rag_pipeline

# Build enhanced RAG system with query rewriting and reranking
enhanced_rag = build_enhanced_rag_pipeline(
    embedding_model="BAAI/bge-small-en-v1.5",
    training_csv="data/training.csv",
    output_dir="data/enhanced_rag",
    llm_model="gpt-4o-mini",
    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Enhanced query with rewriting and reranking
answer = enhanced_rag.enhanced_query("What is the capital of Uruguay?")
print(answer)
```

3. **Run RAGAS evaluation**
```python
from src.ragas_evaluation_clean import RAGASEvaluator
import pandas as pd

# Load test data
test_data = pd.read_csv("data/test.csv")

# Initialize evaluator
evaluator = RAGASEvaluator(test_data, num_samples=5)

# Run evaluation
results = evaluator.run_evaluation()
print(results)
```

## 📊 Dataset Information

### Training Data (`data/training.csv`)
- **Size**: 3,200 Wikipedia passages
- **Average Length**: 389.85 characters
- **Format**: CSV with columns: `passage`, `id`, `passage_length`
- **Source**: Wikipedia articles processed into coherent passages

### Test Data (`data/test.csv`)
- **Size**: 918 question-answer pairs
- **Question Length**: 4-252 characters (average: 53.09)
- **Answer Length**: 1-423 characters (average: 19.18)
- **Format**: CSV with columns: `question`, `answer`, `id`, `answer_length`

## 🔧 Core Components

### 1. Native RAG Pipeline (`src/langchain_native_pipeline.py`)

**Key Features**:
- LangChain-based implementation
- Multiple embedding models support
- FAISS vector store integration
- OpenRouter and Gemini LLM support
- Configurable retrieval parameters

**Supported Models**:
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`
- **LLMs**: GPT-4, GPT-4o-mini, Gemini-2.0-flash-lite, DeepSeek

### 2. Enhanced RAG Features (`src/enhanced_rag_features.py`)

**Advanced Features**:
- **Query Rewriting**: Three strategies (Conservative, Balanced, Aggressive)
- **Document Reranking**: Cross-encoder based relevance scoring
- **Intelligent Strategy Selection**: Automatic complexity analysis
- **Quality Control**: Over-complexity detection and fallback

**Key Classes**:
- `QueryRewriter`: Intelligent query optimization
- `DocumentReranker`: Cross-encoder based reranking
- `EnhancedRAGSystem`: Integrated enhanced RAG system

### 3. RAGAS Evaluation (`src/RAGASCalculator.py`, `src/ragas_evaluation_clean.py`)

**Evaluation Metrics**:
- **Faithfulness**: Measures answer grounding in retrieved context
- **Context Precision**: Evaluates relevance of retrieved context
- **Context Recall**: Assesses coverage of ground truth information
- **Answer Relevance**: Measures direct relevance of generated answers

**Key Features**:
- Asynchronous evaluation for efficiency
- Rate limit handling with retry mechanisms
- Comprehensive error handling and logging
- Support for both Basic and Enhanced RAG systems

## 📈 Experimental Results

### Prompt Strategy Evaluation (Phase 3)
| Strategy | F1-Score | Exact Match | Accuracy |
|----------|----------|-------------|----------|
| **Basic** | **0.7200** | **0.4000** | **0.4000** |
| Few-shot | 0.5467 | 0.6000 | 0.6000 |
| Persona | 0.3800 | 0.6000 | 0.6000 |

**Key Finding**: Basic strategy achieved highest F1-score, demonstrating that simple, direct prompting is most effective for factual question-answering tasks.

### Parameter Optimization (Phase 4)
| Group | Embedding Dim | Model | K Value | F1-Score | Exact Match | Improvement |
|-------|---------------|-------|---------|----------|-------------|-------------|
| Group1 | 384 | all-MiniLM-L6-v2 | 3 | 0.4000 | 0.4000 | Baseline |
| Group1 | 384 | all-MiniLM-L6-v2 | 10 | 0.4571 | 0.4000 | +14.3% |
| Group2 | 512 | bge-base-en-v1.5 | 3 | 0.6250 | 0.6000 | +56.3% |
| Group2 | 512 | bge-base-en-v1.5 | 10 | 0.6571 | 0.6000 | +64.3% |

**Key Finding**: 512D model significantly outperformed 384D model with 50-64% F1-score improvement, demonstrating the importance of high-quality embeddings.

### RAGAS Evaluation (Phase 5-6)
| Metric | Basic RAG | Enhanced RAG | Performance Difference |
|--------|-----------|--------------|----------------------|
| **Faithfulness** | 1.000 | 1.000 | No difference |
| **Context Precision** | 0.70 | 0.70 | No difference |
| **Context Recall** | 0.600 | 0.600 | No difference |
| **Answer Relevance** | 0.534 | 0.424 | -0.110 (Basic RAG better) |

**Key Finding**: Basic RAG system unexpectedly outperformed Enhanced RAG system in answer relevance, highlighting the importance of empirical evaluation.

## 🛠️ Development Phases

### Phase 1: Dataset Setup
- **Deliverable**: `deliverables/phase1_domain documents/deliverable1_dataset_setup.ipynb`
- **Content**: Dataset analysis, quality assessment, and preprocessing
- **Key Insights**: 3,200 training passages, 918 test questions, high data quality

### Phase 2: Naive RAG Implementation
- **Deliverable**: `deliverables/phase2_Naive RAG Implementation/`
- **Content**: Basic RAG system implementation using LangChain
- **Key Features**: FAISS vector store, multiple LLM support, configurable parameters

### Phase 3: Prompt Strategy Evaluation
- **Deliverable**: `deliverables/phase3/step3_evaluation_report.md`
- **Content**: Evaluation of three prompt strategies (Basic, Few-shot, Persona)
- **Key Finding**: Basic strategy achieved highest F1-score (0.72)

### Phase 4: Parameter Optimization
- **Deliverable**: `deliverables/phase4/step4_parameter_analysis_en.md`
- **Content**: Comprehensive evaluation of embedding dimensions and retrieval counts
- **Key Finding**: 512D model with K=10 achieved optimal performance

### Phase 5-6: Enhanced Features and RAGAS Evaluation
- **Deliverable**: `deliverables/phase5-6/`
- **Content**: Query rewriting, document reranking, and RAGAS evaluation
- **Key Finding**: Basic RAG outperformed Enhanced RAG in key metrics

### Phase 7: Final Summary
- **Deliverable**: `deliverables/phase7/final_summary_report.md`
- **Content**: Comprehensive synthesis and reflection (1,247 words)
- **Key Insights**: Empirical testing importance, simplicity advantages, feature integration complexity

## 🔍 Key Insights and Lessons Learned

### 1. Empirical Testing is Essential
Theoretical improvements do not always translate to better performance. The enhanced system's underperformance demonstrates the importance of rigorous empirical evaluation.

### 2. Simplicity Often Wins
The basic RAG system's superior performance suggests that simpler, more direct approaches may be more effective for certain use cases.

### 3. Feature Integration Complexity
Adding multiple enhancements simultaneously can create unexpected interactions that negatively impact performance.

### 4. Embedding Quality is Critical
The 512D model's significant performance improvement over the 384D model demonstrates the importance of high-quality embeddings in RAG systems.

### 5. Sample Size Matters
Limited evaluation samples (5 questions) may not provide statistically significant results, highlighting the need for larger-scale evaluations.

## 🚀 Future Improvements

### Immediate Improvements
1. **Scale Up Evaluation**: Conduct larger-scale evaluations with 100+ samples
2. **Feature Ablation Study**: Systematically test individual enhanced features
3. **Parameter Tuning**: Fine-tune query rewriting and reranking algorithms

### Advanced Enhancements
1. **Metadata Filtering**: Implement document metadata filtering
2. **Multi-Vector Retrieval**: Explore multiple embedding strategies
3. **Confidence Scoring**: Develop answer confidence estimation
4. **Context Window Optimization**: Implement dynamic context management
5. **Grounded Citations**: Add source attribution and verification

## 📚 Dependencies

### Core Dependencies
- **langchain>=0.1.0**: Core RAG framework
- **faiss-cpu>=1.7.4**: Vector similarity search
- **sentence-transformers>=2.2.2**: Embedding models
- **ragas>=0.1.0**: RAGAS evaluation metrics
- **pandas>=1.5.0**: Data processing
- **numpy>=1.24.0**: Numerical computations

### Optional Dependencies
- **google-generativeai**: Google Gemini integration
- **faiss-gpu>=1.7.4**: GPU acceleration (if CUDA available)


## 📄 License

This project is developed for educational and research purposes. Please ensure compliance with API usage terms and conditions for OpenAI, Google, and other services used.

## 🤝 Contributing

This is an academic project. For questions or suggestions, please refer to the project documentation or contact the development team.
