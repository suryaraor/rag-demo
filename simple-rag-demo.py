#!/usr/bin/env python3
"""
Simple RAG Demo - Understanding Retrieval-Augmented Generation
A basic implementation to demonstrate RAG concepts with real code.
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import re
from dataclasses import dataclass
from datetime import datetime

# For this demo, we'll use a simple cosine similarity implementation
# In production, you'd use proper embedding models like sentence-transformers

@dataclass
class Document:
    """Represents a document in our knowledge base"""
    id: str
    title: str
    content: str
    metadata: Dict

class SimpleEmbedding:
    """Simple TF-IDF-like embedding for demonstration"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
    
    def fit(self, documents: List[str]):
        """Build vocabulary and compute IDF scores"""
        # Build vocabulary
        vocab_set = set()
        for doc in documents:
            words = self._tokenize(doc)
            vocab_set.update(words)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(vocab_set))}
        
        # Compute IDF scores
        doc_count = len(documents)
        word_doc_count = {}
        
        for doc in documents:
            words = set(self._tokenize(doc))
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1
        
        for word in self.vocabulary:
            self.idf_scores[word] = np.log(doc_count / (word_doc_count.get(word, 1) + 1))
    
    def encode(self, text: str) -> np.ndarray:
        """Convert text to vector representation"""
        words = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        
        # Compute term frequency
        word_count = {}
        for word in words:
            word_count[word] = word_count.get(word, 0) + 1
        
        # Create TF-IDF vector
        for word, count in word_count.items():
            if word in self.vocabulary:
                idx = self.vocabulary[word]
                tf = count / len(words)
                idf = self.idf_scores.get(word, 0)
                vector[idx] = tf * idf
        
        # Normalize vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on non-alphanumeric characters
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words

class SimpleRAG:
    """A simple RAG implementation for demonstration"""
    
    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: List[np.ndarray] = []
        self.embedding_model = SimpleEmbedding()
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base"""
        print(f"📚 Adding {len(documents)} documents to knowledge base...")
        
        self.documents.extend(documents)
        
        # Extract content for embedding training
        all_content = [doc.content for doc in self.documents]
        
        # Fit embedding model on all documents
        self.embedding_model.fit(all_content)
        
        # Generate embeddings for all documents
        self.embeddings = []
        for doc in self.documents:
            embedding = self.embedding_model.encode(doc.content)
            self.embeddings.append(embedding)
        
        print(f"✅ Knowledge base now contains {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """Retrieve most relevant documents for a query"""
        print(f"🔍 Searching for: '{query}'")
        
        if not self.documents:
            print("❌ No documents in knowledge base!")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query)
        
        # Compute similarities
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((self.documents[i], similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"📊 Found {len(top_results)} relevant documents")
        for i, (doc, score) in enumerate(top_results):
            print(f"  {i+1}. {doc.title} (similarity: {score:.3f})")
        
        return top_results
    
    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate an answer using retrieved context (simulated LLM)"""
        print("🤖 Generating answer...")
        
        if not context_docs:
            return "I couldn't find relevant information to answer your question."
        
        # In a real RAG system, this would call an LLM like GPT-4, Claude, etc.
        # Here we'll simulate it with rule-based responses
        
        context = "\n\n".join([f"From {doc.title}: {doc.content}" for doc in context_docs])
        
        # Simple response generation based on query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['vacation', 'time off', 'leave']):
            return self._generate_vacation_response(context_docs)
        elif any(word in query_lower for word in ['expense', 'report', 'reimbursement']):
            return self._generate_expense_response(context_docs)
        elif any(word in query_lower for word in ['product', 'service', 'offer']):
            return self._generate_product_response(context_docs)
        elif any(word in query_lower for word in ['hour', 'schedule', 'work']):
            return self._generate_schedule_response(context_docs)
        else:
            # Generic response using first document
            doc = context_docs[0]
            return f"Based on our {doc.title}, {doc.content[:200]}..."
    
    def _generate_vacation_response(self, docs: List[Document]) -> str:
        """Generate vacation policy response"""
        for doc in docs:
            if 'vacation' in doc.content.lower() or 'leave' in doc.content.lower():
                return f"According to our vacation policy: {doc.content} (Source: {doc.title})"
        return "I found some information but couldn't locate specific vacation policy details."
    
    def _generate_expense_response(self, docs: List[Document]) -> str:
        """Generate expense policy response"""
        for doc in docs:
            if 'expense' in doc.content.lower() or 'reimbursement' in doc.content.lower():
                return f"For expense reporting: {doc.content} (Source: {doc.title})"
        return "I found some information but couldn't locate specific expense reporting details."
    
    def _generate_product_response(self, docs: List[Document]) -> str:
        """Generate product information response"""
        for doc in docs:
            if 'product' in doc.content.lower() or 'service' in doc.content.lower():
                return f"Our product offerings: {doc.content} (Source: {doc.title})"
        return "I found some information but couldn't locate specific product details."
    
    def _generate_schedule_response(self, docs: List[Document]) -> str:
        """Generate work schedule response"""
        for doc in docs:
            if 'hour' in doc.content.lower() or 'schedule' in doc.content.lower():
                return f"Regarding work schedule: {doc.content} (Source: {doc.title})"
        return "I found some information but couldn't locate specific schedule details."
    
    def answer_question(self, query: str, top_k: int = 3) -> Dict:
        """Main RAG pipeline: retrieve + generate"""
        print(f"\n{'='*60}")
        print(f"❓ Question: {query}")
        print(f"{'='*60}")
        
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k)
        
        if not retrieved_docs:
            return {
                "answer": "I don't have enough information to answer that question.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Step 2: Extract documents and calculate confidence
        relevant_docs = [doc for doc, score in retrieved_docs]
        confidence = retrieved_docs[0][1] if retrieved_docs else 0.0
        
        # Step 3: Generate answer
        answer = self.generate_answer(query, relevant_docs)
        
        result = {
            "answer": answer,
            "sources": [{"title": doc.title, "score": score} for doc, score in retrieved_docs],
            "confidence": confidence
        }
        
        print(f"\n💡 Answer: {answer}")
        print(f"📚 Sources used: {len(relevant_docs)}")
        print(f"🎯 Confidence: {confidence:.3f}")
        
        return result

def create_sample_knowledge_base() -> List[Document]:
    """Create sample documents for demonstration"""
    documents = [
        Document(
            id="doc_001",
            title="Employee Vacation Policy",
            content="All full-time employees are entitled to 20 vacation days per year. Part-time employees receive prorated vacation based on hours worked. Vacation requests must be submitted at least 2 weeks in advance through the HR portal. Unused vacation days can be carried over to the next year, up to a maximum of 5 days. Extended leave requests (more than 2 weeks) require manager and HR approval.",
            metadata={"category": "HR", "last_updated": "2024-01-15"}
        ),
        Document(
            id="doc_002",
            title="Expense Reporting Guidelines",
            content="Monthly expense reports are due by the 5th of the following month. All receipts must be attached and categorized properly. Business meal allowance is $75 per day during travel. Hotel expenses over $200/night require pre-approval. Reimbursements are processed within 5 business days of approval. Use the company expense management system for all submissions.",
            metadata={"category": "Finance", "last_updated": "2024-02-01"}
        ),
        Document(
            id="doc_003",
            title="Product Catalog Overview",
            content="Our company offers three main product lines: CloudSync Pro for data synchronization starting at $29/month, SecureVault for encrypted cloud storage starting at $19/month, and AnalyticsDash for business intelligence starting at $49/month. All products include 24/7 customer support, regular security updates, and enterprise-grade features. Custom enterprise solutions are available for large organizations.",
            metadata={"category": "Products", "last_updated": "2024-01-30"}
        ),
        Document(
            id="doc_004",
            title="Work Schedule and Remote Work Policy",
            content="Standard office hours are 9:00 AM to 5:00 PM, Monday through Friday. Core collaboration hours when all team members should be available are 10:00 AM to 3:00 PM. Remote work is permitted up to 3 days per week with manager approval. Flexible start times between 8:00 AM and 10:00 AM are allowed. All meetings should be scheduled during core hours when possible.",
            metadata={"category": "HR", "last_updated": "2024-01-20"}
        ),
        Document(
            id="doc_005",
            title="IT Security Guidelines",
            content="All employees must use strong passwords with at least 12 characters including numbers and symbols. Two-factor authentication is required for all company systems. Personal devices used for work must have approved security software installed. USB drives are not permitted without IT approval. Report any security incidents immediately to the IT help desk.",
            metadata={"category": "IT", "last_updated": "2024-02-10"}
        ),
        Document(
            id="doc_006",
            title="Meeting Room Booking Procedures",
            content="Meeting rooms can be booked through the company calendar system up to 30 days in advance. Bookings for more than 2 hours require manager approval. Conference room equipment includes projectors, whiteboards, and video conferencing capabilities. Clean up after meetings and report any equipment issues. Cancel unused bookings to allow others to use the space.",
            metadata={"category": "Facilities", "last_updated": "2024-01-25"}
        )
    ]
    return documents

def run_interactive_demo():
    """Run an interactive RAG demo"""
    print("🧠 Simple RAG Demo")
    print("=" * 50)
    print("This demo shows how Retrieval-Augmented Generation works!")
    print()
    
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Add sample documents
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    print("\n🎯 Available topics in knowledge base:")
    for doc in documents:
        print(f"  • {doc.title}")
    
    print("\n" + "=" * 50)
    print("Try asking questions! (type 'quit' to exit)")
    print("Example questions:")
    print("  • What is the vacation policy?")
    print("  • How do I submit expense reports?")
    print("  • What products does the company offer?")
    print("  • What are the working hours?")
    print("=" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the RAG demo!")
                break
            
            if not question:
                print("Please enter a question.")
                continue
            
            # Process question with RAG
            result = rag.answer_question(question)
            
            print(f"\n📝 Sources consulted:")
            for source in result["sources"]:
                print(f"  • {source['title']} (relevance: {source['score']:.3f})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for trying the RAG demo!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def run_batch_demo():
    """Run demo with predefined questions"""
    print("🧠 RAG Demo - Batch Processing")
    print("=" * 50)
    
    # Initialize RAG system
    rag = SimpleRAG()
    documents = create_sample_knowledge_base()
    rag.add_documents(documents)
    
    # Test questions
    test_questions = [
        "What is the company's vacation policy?",
        "How do I submit an expense report?",
        "What products does the company offer?",
        "What are the working hours?",
        "How do I book a meeting room?",
        "What are the IT security requirements?",
        "Can I work remotely?",
        "What is the meal allowance for business travel?"
    ]
    
    print(f"\n🧪 Testing {len(test_questions)} questions...")
    
    results = []
    for question in test_questions:
        result = rag.answer_question(question)
        results.append((question, result))
    
    print("\n📊 Summary of Results:")
    print("=" * 50)
    for question, result in results:
        print(f"Q: {question}")
        print(f"A: {result['answer'][:100]}...")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 30)

if __name__ == "__main__":
    print("🚀 Choose demo mode:")
    print("1. Interactive mode (ask your own questions)")
    print("2. Batch mode (see predefined examples)")
    
    while True:
        choice = input("\nEnter choice (1 or 2): ").strip()
        if choice == "1":
            run_interactive_demo()
            break
        elif choice == "2":
            run_batch_demo()
            break
        else:
            print("Please enter 1 or 2")
