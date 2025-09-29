# Naive RAG System - Code Organization

This directory contains a modular implementation of a Naive RAG (Retrieval-Augmented Generation) system with clear documentation, error handling, and configuration management.

## 📁 File Structure

```
src/
├── naive.py          # Main RAG system implementation
├── config.py         # Configuration management
├── utils.py          # Utility functions
├── run_rag.py        # Command-line interface
└── README.md         # This file
```

## 🔧 Modules Overview

### 1. `naive.py` - Core RAG System
**Purpose**: Main implementation of the RAG system

**Key Features**:
- `NaiveRAGSystem` class with full functionality
- Comprehensive error handling and logging
- Modular design with clear separation of concerns
- Type hints for better code documentation

**Main Methods**:
- `load_training_data()` - Load training passages
- `load_test_data()` - Load test questions and answers
- `initialize_model()` - Initialize sentence transformer
- `create_embeddings()` - Generate embeddings
- `create_index()` - Build FAISS index
- `search_similar_passages()` - Retrieve relevant passages
- `evaluate_retrieval_accuracy()` - Evaluate system performance
- `test_single_query()` - Test individual queries
- `test_sample_questions()` - Test multiple questions
- `run_full_evaluation()` - Complete system evaluation

### 2. `config.py` - Configuration Management
**Purpose**: Centralized configuration management

**Key Features**:
- Environment-specific configurations
- Easy parameter adjustment
- Default values for all parameters
- Configuration validation

**Configuration Classes**:
- `Config` - Default configuration
- `DevelopmentConfig` - Development environment
- `ProductionConfig` - Production environment
- `TestingConfig` - Testing environment

**Key Parameters**:
- Model settings (model name, embedding batch size)
- Retrieval settings (number of passages to retrieve)
- Data paths (training and test data locations)
- Logging settings (log level, directory)
- Evaluation settings (number of samples)

### 3. `utils.py` - Utility Functions
**Purpose**: Common utility functions and helpers

**Key Features**:
- File validation and operations
- Data quality validation
- Text processing utilities
- System information gathering
- Environment setup

**Main Functions**:
- `validate_file_path()` - Validate file existence and format
- `create_directory_if_not_exists()` - Directory management
- `save_results_to_file()` - Save results to JSON
- `calculate_text_similarity()` - Simple text similarity
- `validate_data_quality()` - Data quality checks
- `setup_environment()` - Environment initialization

### 4. `run_rag.py` - Command-Line Interface
**Purpose**: Command-line interface for running the system

**Key Features**:
- Argument parsing with help text
- Multiple execution modes
- Configuration override options
- Results saving capabilities

**Usage Examples**:
```bash
# Basic usage
python run_rag.py

# Development environment
python run_rag.py --environment development

# Custom parameters
python run_rag.py --k 5 --samples 20

# Single query test
python run_rag.py --query "What is the capital of France?"

# Save results
python run_rag.py --save-results results.json
```

## 🚀 Quick Start

### 1. Basic Usage
```python
from naive import NaiveRAGSystem

# Initialize system
rag_system = NaiveRAGSystem()

# Run full evaluation
rag_system.run_full_evaluation()
```

### 2. Custom Configuration
```python
from naive import NaiveRAGSystem
from config import get_config

# Get custom configuration
config = get_config('development')
config['k'] = 5  # Retrieve 5 passages

# Initialize with custom config
rag_system = NaiveRAGSystem(config)
rag_system.run_full_evaluation()
```

### 3. Command Line Usage
```bash
# Run with default settings
python run_rag.py

# Run with custom parameters
python run_rag.py --k 3 --samples 15 --environment development

# Test single query
python run_rag.py --query "What is machine learning?"
```

## 📋 Configuration Options

### Model Configuration
- `model_name`: Sentence transformer model to use
- `k`: Number of passages to retrieve
- `embedding_batch_size`: Batch size for embedding generation

### Data Configuration
- `training_data_path`: Path to training data CSV
- `test_data_path`: Path to test data CSV

### Logging Configuration
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_dir`: Directory for log files

### Evaluation Configuration
- `evaluation_samples`: Number of samples for evaluation
- `test_samples`: Number of samples for testing

## 🔍 Error Handling

The system includes comprehensive error handling:

1. **File Operations**: Validates file existence and format
2. **Data Loading**: Checks for required columns and data quality
3. **Model Operations**: Handles model initialization and embedding generation
4. **Index Operations**: Manages FAISS index creation and search
5. **System Operations**: Logs all operations and errors

## 📊 Logging

The system provides detailed logging:

- **File Logging**: All operations logged to timestamped files
- **Console Logging**: Real-time output for monitoring
- **Error Logging**: Detailed error information for debugging
- **Performance Logging**: Timing and resource usage information

## 🧪 Testing

### Unit Testing
```python
# Test individual components
from naive import NaiveRAGSystem

rag_system = NaiveRAGSystem()
rag_system.setup_system()

# Test single query
results = rag_system.test_single_query("Test query")
```

### Integration Testing
```python
# Test complete workflow
rag_system.run_full_evaluation()
```

## 📈 Performance Monitoring

The system includes performance monitoring:

- **Embedding Generation**: Tracks time and memory usage
- **Index Creation**: Monitors index building performance
- **Search Operations**: Measures retrieval speed
- **Evaluation**: Tracks accuracy and performance metrics

## 🔧 Customization

### Adding New Models
```python
# Update configuration
config = {
    'model_name': 'your-custom-model',
    'k': 10,
    # ... other parameters
}

rag_system = NaiveRAGSystem(config)
```

### Custom Evaluation Metrics
```python
# Override evaluation method
class CustomRAGSystem(NaiveRAGSystem):
    def evaluate_retrieval_accuracy(self, num_samples=10):
        # Custom evaluation logic
        pass
```

## 📝 Documentation

- **Type Hints**: All functions include type annotations
- **Docstrings**: Comprehensive documentation for all methods
- **Examples**: Usage examples in docstrings
- **Error Handling**: Detailed error messages and logging

## 🚀 Future Enhancements

- **Answer Generation**: Add LLM integration for answer generation
- **Advanced Metrics**: Implement more sophisticated evaluation metrics
- **Caching**: Add embedding and index caching
- **Distributed Processing**: Support for distributed embedding generation
- **API Interface**: REST API for system interaction
