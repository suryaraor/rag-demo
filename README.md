# RAG Demo - Understanding Retrieval-Augmented Generation

A comprehensive demo to understand RAG (Retrieval-Augmented Generation) concepts with interactive examples.

## 🎯 What is RAG?

RAG (Retrieval-Augmented Generation) is an AI technique that combines:
1. **Retrieval**: Finding relevant information from a knowledge base
2. **Augmentation**: Adding that information to the AI's context
3. **Generation**: Creating responses based on both training and retrieved data

## 🌟 Why RAG is Revolutionary

Without RAG:
- AI responses are limited to training data
- No access to current/private information
- Can't answer questions about your specific documents

With RAG:
- AI can access your latest documents
- Provides accurate, source-backed answers
- Reduces hallucinations significantly

## 📁 Demo Structure

```
rag-demo/
├── README.md                 # This file
├── index.html               # Interactive web demo
├── simple-rag-demo.py       # Python implementation
├── advanced-rag-demo.py     # Advanced features
├── data/                    # Sample documents
│   ├── company-docs/        # Sample company documents
│   └── knowledge-base/      # General knowledge documents
├── embeddings/              # Vector storage
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

1. **Web Demo (No Setup Required)**
   ```bash
   # Open index.html in your browser
   # Interactive visual demonstration
   ```

2. **Python Demo**
   ```bash
   pip install -r requirements.txt
   python simple-rag-demo.py
   ```

3. **Advanced Demo**
   ```bash
   python advanced-rag-demo.py
   ```

## 🎓 Learning Path

1. **Concept Understanding**: Start with the web demo
2. **Basic Implementation**: Run simple-rag-demo.py
3. **Advanced Features**: Explore advanced-rag-demo.py
4. **Customization**: Add your own documents

## 📚 What You'll Learn

- ✅ RAG fundamentals and workflow
- ✅ Document chunking strategies
- ✅ Vector embeddings and similarity search
- ✅ Prompt engineering for RAG
- ✅ Evaluation and improvement techniques
- ✅ Real-world implementation patterns

## 🛠️ Features Demonstrated

- Document ingestion and preprocessing
- Text chunking and embedding generation
- Vector similarity search
- Context injection and prompt construction
- Response generation with source attribution
- Interactive Q&A interface

Ready to dive into RAG? Start with the interactive demo!
