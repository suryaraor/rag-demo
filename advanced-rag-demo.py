#!/usr/bin/env python3
"""
Advanced RAG Demo - Production-Ready Features
Demonstrates advanced RAG concepts with real embedding models and vector databases.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

# For production RAG, you'd typically use these libraries:
# pip install sentence-transformers faiss-cpu openai chromadb

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Using simple embeddings.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Using numpy similarity search.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Enhanced document representation"""
    id: str
    title: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()

@dataclass
class RetrievalResult:
    """Structured retrieval result"""
    document: Document
    similarity_score: float
    rank: int

@dataclass
class RAGResponse:
    """Comprehensive RAG response"""
    query: str
    answer: str
    sources: List[RetrievalResult]
    confidence: float
    processing_time: float
    metadata: Dict

class AdvancedEmbedding:
    """Advanced embedding model using sentence-transformers or fallback"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            print(f"🤖 Loading sentence transformer: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            print(f"✅ Model loaded. Embedding dimension: {self.embedding_dim}")
        else:
            print("📝 Using simple TF-IDF embeddings (install sentence-transformers for better results)")
            self.model = None
            self.vocabulary = {}
            self.idf_scores = {}
            self.embedding_dim = 384  # Default dimension
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings"""
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            return self.model.encode(texts, convert_to_numpy=True)
        else:
            return self._simple_encode(texts)
    
    def _simple_encode(self, texts: List[str]) -> np.ndarray:
        """Fallback simple encoding"""
        # This is a simplified version - in practice you'd want a more sophisticated approach
        embeddings = []
        for text in texts:
            # Create a simple hash-based embedding
            words = text.lower().split()
            embedding = np.zeros(self.embedding_dim)
            for i, word in enumerate(words[:self.embedding_dim]):
                embedding[i % self.embedding_dim] += hash(word) % 100 / 100.0
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            embeddings.append(embedding)
        
        return np.array(embeddings)

class VectorStore:
    """Vector storage and similarity search"""
    
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.documents: List[Document] = []
        
        if FAISS_AVAILABLE:
            print("🗄️  Using FAISS for vector storage")
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product (cosine similarity)
            self.use_faiss = True
        else:
            print("📊 Using numpy for vector storage")
            self.embeddings_matrix = None
            self.use_faiss = False
    
    def add_documents(self, documents: List[Document]):
        """Add documents with their embeddings to the store"""
        if not documents:
            return
        
        self.documents.extend(documents)
        embeddings = np.array([doc.embedding for doc in documents])
        
        if self.use_faiss:
            # Normalize for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
        else:
            # Stack embeddings for numpy search
            if self.embeddings_matrix is None:
                self.embeddings_matrix = embeddings
            else:
                self.embeddings_matrix = np.vstack([self.embeddings_matrix, embeddings])
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[RetrievalResult]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []
        
        query_embedding = query_embedding.reshape(1, -1)
        
        if self.use_faiss:
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
            similarities, indices = self.index.search(query_embedding, min(top_k, len(self.documents)))
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Numpy-based search
            similarities = np.dot(self.embeddings_matrix, query_embedding.T).flatten()
            indices = np.argsort(similarities)[::-1][:top_k]
            similarities = similarities[indices]
        
        results = []
        for rank, (idx, score) in enumerate(zip(indices, similarities)):
            if idx < len(self.documents):  # Ensure valid index
                result = RetrievalResult(
                    document=self.documents[idx],
                    similarity_score=float(score),
                    rank=rank + 1
                )
                results.append(result)
        
        return results

class DocumentProcessor:
    """Advanced document processing and chunking"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def process_document(self, doc_id: str, title: str, content: str, metadata: Dict = None) -> List[Document]:
        """Process a document into chunks"""
        if metadata is None:
            metadata = {}
        
        # Clean content
        cleaned_content = self._clean_text(content)
        
        # Create chunks
        chunks = self._create_chunks(cleaned_content)
        
        # Create document objects
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_metadata = {
                **metadata,
                "parent_doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            }
            
            doc = Document(
                id=chunk_id,
                title=f"{title} (Part {i+1})",
                content=chunk,
                metadata=chunk_metadata
            )
            documents.append(doc)
        
        return documents
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Basic cleaning - in production you'd want more sophisticated cleaning
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = ' '.join(text.split())  # Normalize whitespace
        return text
    
    def _create_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            if i + self.chunk_size >= len(words):
                break
        
        return chunks

class AdvancedRAG:
    """Production-ready RAG system"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = AdvancedEmbedding(embedding_model)
        self.vector_store = VectorStore(self.embedding_model.embedding_dim)
        self.document_processor = DocumentProcessor()
        self.query_history: List[Dict] = []
        
        # Create storage directory
        self.storage_dir = Path("./rag_storage")
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize metadata database
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """Initialize SQLite database for metadata"""
        db_path = self.storage_dir / "metadata.db"
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT,
                content TEXT,
                metadata TEXT,
                created_at TEXT,
                embedding_model TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                response TEXT,
                timestamp TEXT,
                confidence REAL,
                sources TEXT
            )
        """)
        
        self.conn.commit()
    
    def add_document(self, doc_id: str, title: str, content: str, metadata: Dict = None) -> int:
        """Add a document to the RAG system"""
        logger.info(f"Processing document: {title}")
        
        # Process document into chunks
        documents = self.document_processor.process_document(doc_id, title, content, metadata)
        
        # Generate embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Store in vector database
        self.vector_store.add_documents(documents)
        
        # Store metadata in SQLite
        for doc in documents:
            self.conn.execute("""
                INSERT OR REPLACE INTO documents 
                (id, title, content, metadata, created_at, embedding_model)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc.id, doc.title, doc.content, 
                json.dumps(doc.metadata), doc.created_at,
                self.embedding_model.model_name
            ))
        
        self.conn.commit()
        logger.info(f"Added {len(documents)} chunks from document: {title}")
        return len(documents)
    
    def retrieve(self, query: str, top_k: int = 5, min_similarity: float = 0.1) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Search vector store
        results = self.vector_store.search(query_embedding, top_k)
        
        # Filter by minimum similarity
        filtered_results = [r for r in results if r.similarity_score >= min_similarity]
        
        logger.info(f"Retrieved {len(filtered_results)} relevant documents for query: {query[:50]}...")
        return filtered_results
    
    def generate_answer(self, query: str, context_results: List[RetrievalResult]) -> str:
        """Generate answer using retrieved context"""
        if not context_results:
            return "I couldn't find relevant information to answer your question."
        
        # In production, this would call a real LLM (GPT-4, Claude, etc.)
        # Here we'll create a sophisticated rule-based response
        
        context = "\n\n".join([
            f"Source {i+1} ({result.document.title}): {result.document.content[:300]}..."
            for i, result in enumerate(context_results[:3])
        ])
        
        # Analyze query intent
        query_lower = query.lower()
        
        # Generate contextual response based on content
        if self._contains_keywords(query_lower, ['vacation', 'leave', 'time off']):
            return self._generate_policy_response("vacation", context_results)
        elif self._contains_keywords(query_lower, ['expense', 'reimbursement', 'cost']):
            return self._generate_policy_response("expense", context_results)
        elif self._contains_keywords(query_lower, ['product', 'service', 'offer', 'solution']):
            return self._generate_product_response(context_results)
        elif self._contains_keywords(query_lower, ['schedule', 'hours', 'work', 'remote']):
            return self._generate_schedule_response(context_results)
        elif self._contains_keywords(query_lower, ['meeting', 'room', 'book', 'reserve']):
            return self._generate_facility_response(context_results)
        else:
            return self._generate_general_response(context_results)
    
    def _contains_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if text contains any of the keywords"""
        return any(keyword in text for keyword in keywords)
    
    def _generate_policy_response(self, policy_type: str, results: List[RetrievalResult]) -> str:
        """Generate policy-specific response"""
        relevant_result = None
        for result in results:
            if policy_type in result.document.content.lower():
                relevant_result = result
                break
        
        if relevant_result:
            content = relevant_result.document.content
            source = relevant_result.document.title
            confidence = relevant_result.similarity_score
            
            return f"According to our {policy_type} policy: {content}\n\nSource: {source} (confidence: {confidence:.2f})"
        else:
            return f"I found some information but couldn't locate specific {policy_type} policy details."
    
    def _generate_product_response(self, results: List[RetrievalResult]) -> str:
        """Generate product information response"""
        for result in results:
            if any(word in result.document.content.lower() for word in ['product', 'service', 'offer']):
                return f"Our products and services: {result.document.content}\n\nSource: {result.document.title}"
        return "I found some information but couldn't locate specific product details."
    
    def _generate_schedule_response(self, results: List[RetrievalResult]) -> str:
        """Generate schedule/work hours response"""
        for result in results:
            if any(word in result.document.content.lower() for word in ['hour', 'schedule', 'remote']):
                return f"Regarding work schedule: {result.document.content}\n\nSource: {result.document.title}"
        return "I found some information but couldn't locate specific schedule details."
    
    def _generate_facility_response(self, results: List[RetrievalResult]) -> str:
        """Generate facility/meeting room response"""
        for result in results:
            if any(word in result.document.content.lower() for word in ['meeting', 'room', 'book']):
                return f"For meeting room information: {result.document.content}\n\nSource: {result.document.title}"
        return "I found some information but couldn't locate specific facility details."
    
    def _generate_general_response(self, results: List[RetrievalResult]) -> str:
        """Generate general response using best match"""
        if results:
            best_result = results[0]
            return f"Based on the available information: {best_result.document.content[:400]}...\n\nSource: {best_result.document.title}"
        return "I couldn't find relevant information to answer your question."
    
    def answer_question(self, query: str, top_k: int = 5) -> RAGResponse:
        """Complete RAG pipeline"""
        start_time = datetime.now()
        
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant documents
        retrieval_results = self.retrieve(query, top_k)
        
        # Calculate confidence based on top result
        confidence = retrieval_results[0].similarity_score if retrieval_results else 0.0
        
        # Generate answer
        answer = self.generate_answer(query, retrieval_results)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Create response
        response = RAGResponse(
            query=query,
            answer=answer,
            sources=retrieval_results,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                "total_sources": len(retrieval_results),
                "embedding_model": self.embedding_model.model_name,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Store query in history
        self._store_query(response)
        
        return response
    
    def _store_query(self, response: RAGResponse):
        """Store query and response in database"""
        sources_json = json.dumps([
            {"title": r.document.title, "score": r.similarity_score}
            for r in response.sources
        ])
        
        self.conn.execute("""
            INSERT INTO queries (query, response, timestamp, confidence, sources)
            VALUES (?, ?, ?, ?, ?)
        """, (
            response.query, response.answer, response.metadata["timestamp"],
            response.confidence, sources_json
        ))
        self.conn.commit()
    
    def get_query_history(self, limit: int = 10) -> List[Dict]:
        """Get recent query history"""
        cursor = self.conn.execute("""
            SELECT query, response, timestamp, confidence, sources
            FROM queries
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "query": row[0],
                "response": row[1],
                "timestamp": row[2],
                "confidence": row[3],
                "sources": json.loads(row[4])
            })
        
        return history
    
    def get_document_stats(self) -> Dict:
        """Get statistics about the document collection"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cursor.fetchone()[0]
        
        cursor = self.conn.execute("SELECT COUNT(*) FROM queries")
        query_count = cursor.fetchone()[0]
        
        return {
            "total_documents": doc_count,
            "total_queries": query_count,
            "embedding_dimension": self.embedding_model.embedding_dim,
            "embedding_model": self.embedding_model.model_name
        }

def create_advanced_knowledge_base() -> List[Dict]:
    """Create comprehensive sample documents"""
    documents = [
        {
            "id": "policy_001",
            "title": "Employee Vacation and Leave Policy",
            "content": """
            Our comprehensive vacation and leave policy ensures work-life balance for all employees. 
            
            VACATION ENTITLEMENT:
            - Full-time employees: 20 vacation days per year
            - Part-time employees: Prorated based on hours worked (minimum 15 hours/week)
            - New employees: 1.67 days per month during first year
            - Vacation accrual begins on start date
            
            REQUESTING TIME OFF:
            - Submit requests at least 2 weeks in advance through HR portal
            - Manager approval required for all requests
            - Blackout periods: December 20-31, fiscal year-end weeks
            - Maximum consecutive vacation: 2 weeks without special approval
            
            CARRYOVER POLICY:
            - Up to 5 days can be carried over to next year
            - Carryover days must be used by March 31
            - No cash payout for unused vacation
            
            SICK LEAVE:
            - 10 sick days per year (reset January 1)
            - Doctor's note required for absences >3 consecutive days
            - Family sick leave: up to 3 days per year
            
            EXTENDED LEAVE:
            - Medical leave: up to 12 weeks (FMLA)
            - Parental leave: 6 weeks paid, additional 6 weeks unpaid
            - Sabbatical: available after 7 years of service
            """,
            "metadata": {"category": "HR", "department": "Human Resources", "last_updated": "2024-01-15"}
        },
        {
            "id": "finance_001",
            "title": "Comprehensive Expense Reporting and Reimbursement Policy",
            "content": """
            Our expense reporting system ensures timely reimbursement while maintaining financial controls.
            
            SUBMISSION REQUIREMENTS:
            - Monthly submission by 5th of following month
            - Use company expense management platform (Expensify)
            - All receipts must be attached and legible
            - Business purpose required for all expenses
            - Manager approval needed before reimbursement
            
            TRAVEL EXPENSES:
            - Airfare: Economy class for domestic, business class for international >6 hours
            - Hotels: Up to $200/night in major cities, $150/night elsewhere
            - Meals: $75/day during travel (receipts required >$25)
            - Ground transportation: Reasonable taxi, rideshare, or rental car
            - Mileage: $0.56 per mile for personal vehicle use
            
            BUSINESS MEALS:
            - Client entertainment: Up to $100/person
            - Team meals: Pre-approval required for >$200 total
            - Alcohol: Limited to reasonable amounts during client entertainment
            
            OFFICE SUPPLIES:
            - Individual purchases <$100: No pre-approval needed
            - Office equipment >$100: Manager approval required
            - Software licenses: IT department approval required
            
            PROCESSING:
            - Reimbursement within 5 business days of approval
            - Direct deposit to employee bank account
            - Questions? Contact finance@company.com
            """,
            "metadata": {"category": "Finance", "department": "Finance", "last_updated": "2024-02-01"}
        },
        {
            "id": "product_001",
            "title": "Complete Product Portfolio and Services Catalog",
            "content": """
            Our comprehensive product suite serves businesses of all sizes with cutting-edge technology solutions.
            
            CLOUDSYNV PRO - Data Synchronization Platform
            - Real-time data sync across multiple systems
            - Support for 50+ data sources (databases, APIs, files)
            - Advanced conflict resolution and data validation
            - Pricing: $29/month (Starter), $99/month (Professional), $299/month (Enterprise)
            - Features: 24/7 monitoring, automated backups, custom transformations
            - Use cases: Multi-system integration, data warehousing, real-time analytics
            
            SECUREVAULT - Enterprise Cloud Storage
            - Military-grade encryption (AES-256)
            - Compliance: SOC2, HIPAA, GDPR ready
            - Unlimited versioning and audit trails
            - Pricing: $19/month (100GB), $49/month (1TB), $149/month (Unlimited)
            - Features: Advanced sharing controls, DLP, mobile apps
            - Integrations: Office 365, Google Workspace, Slack
            
            ANALYTICSDASH - Business Intelligence Platform
            - Drag-and-drop dashboard builder
            - 200+ pre-built connectors
            - Machine learning-powered insights
            - Pricing: $49/month (Basic), $149/month (Professional), $399/month (Enterprise)
            - Features: Real-time alerts, custom reports, predictive analytics
            - Support: Live chat, phone, dedicated success manager (Enterprise)
            
            ENTERPRISE SOLUTIONS:
            - Custom development and integration services
            - On-premise deployment options
            - White-label solutions available
            - Volume discounts for 100+ users
            - Professional services: Implementation, training, ongoing support
            
            ALL PLANS INCLUDE:
            - 24/7 customer support
            - 99.9% uptime SLA
            - Regular security updates
            - API access
            - Single sign-on (SSO)
            """,
            "metadata": {"category": "Products", "department": "Product Management", "last_updated": "2024-01-30"}
        },
        {
            "id": "hr_002",
            "title": "Flexible Work Arrangements and Remote Work Policy",
            "content": """
            Our progressive work policy supports productivity and work-life balance through flexible arrangements.
            
            STANDARD WORK SCHEDULE:
            - Office hours: 9:00 AM - 5:00 PM, Monday-Friday
            - Core collaboration hours: 10:00 AM - 3:00 PM (all team members available)
            - Lunch break: 12:00 PM - 1:00 PM (flexible timing allowed)
            - Total work week: 40 hours
            
            FLEXIBLE START TIMES:
            - Permitted range: 8:00 AM - 10:00 AM start times
            - Must maintain 8-hour work day
            - Consistent schedule preferred (±30 minutes)
            - Team coordination required for customer-facing roles
            
            REMOTE WORK POLICY:
            - Up to 3 days per week remote work permitted
            - Manager approval required initially
            - Quarterly performance review for remote work continuation
            - Home office requirements: Reliable internet, quiet workspace
            - Equipment provided: Laptop, monitor, ergonomic accessories
            
            HYBRID WORK GUIDELINES:
            - Minimum 2 days in office per week
            - Tuesday-Thursday recommended for team collaboration
            - All-hands meetings require in-person attendance
            - Client meetings: in-person when possible
            
            PRODUCTIVITY EXPECTATIONS:
            - Maintain same output quality and deadlines
            - Responsive during core hours regardless of location
            - Weekly check-ins with manager
            - Use company communication tools (Slack, Zoom, email)
            
            SPECIAL ARRANGEMENTS:
            - Compressed work weeks (4x10) available in some roles
            - Job sharing opportunities considered case-by-case
            - Sabbatical programs for long-term employees
            """,
            "metadata": {"category": "HR", "department": "Human Resources", "last_updated": "2024-01-20"}
        },
        {
            "id": "it_001",
            "title": "Information Technology Security and Usage Policy",
            "content": """
            Our comprehensive IT security policy protects company and customer data while enabling productive work.
            
            PASSWORD REQUIREMENTS:
            - Minimum 12 characters with numbers, symbols, uppercase, lowercase
            - Unique passwords for each system (no reuse)
            - Password manager required and provided (1Password)
            - Change passwords immediately if compromise suspected
            - No sharing of passwords or accounts
            
            MULTI-FACTOR AUTHENTICATION:
            - Required for all company systems and cloud services
            - Hardware tokens provided for critical systems
            - Backup authentication methods must be configured
            - Report lost/stolen devices immediately to IT
            
            DEVICE SECURITY:
            - All devices must have approved antivirus/EDR software
            - Automatic screen locks after 5 minutes of inactivity
            - Full disk encryption required on laptops and mobile devices
            - Regular security updates and patches mandatory
            - Personal device usage requires MDM enrollment
            
            DATA PROTECTION:
            - Customer data is confidential and protected under NDA
            - No downloading customer data to personal devices
            - Use company-approved cloud storage only
            - Data classification: Public, Internal, Confidential, Restricted
            - Data retention policies apply - contact IT for guidance
            
            NETWORK SECURITY:
            - VPN required for all remote access
            - Guest network available for visitors
            - No unauthorized wireless access points
            - Report suspicious network activity immediately
            
            INCIDENT REPORTING:
            - Report security incidents within 1 hour to security@company.com
            - Include: What happened, when, what data/systems affected
            - Don't attempt to "fix" security incidents yourself
            - Mandatory security training quarterly
            
            ACCEPTABLE USE:
            - Professional use of company resources
            - Limited personal use permitted (lunch, breaks)
            - No illegal downloads or prohibited content
            - Social media use should not reflect negatively on company
            """,
            "metadata": {"category": "IT", "department": "Information Technology", "last_updated": "2024-02-10"}
        },
        {
            "id": "facilities_001",
            "title": "Office Facilities and Meeting Room Management",
            "content": """
            Our modern office facilities support collaboration and productivity with state-of-the-art amenities.
            
            MEETING ROOM BOOKING:
            - Reserve through company calendar system (Outlook/Google)
            - Booking window: Up to 30 days in advance
            - Maximum booking: 4 hours without special approval
            - Recurring meetings: Manager approval for weekly/daily bookings
            - Cancellation: Required 2 hours before meeting start
            
            AVAILABLE ROOMS:
            - Conference Room A: 12 people, video conferencing, projector
            - Conference Room B: 8 people, whiteboard, TV display
            - Huddle Rooms (4): 4 people each, phone/video capable
            - Board Room: 20 people, executive presentations, catering setup
            - Phone Booths (6): Individual quiet calls
            
            ROOM EQUIPMENT:
            - All rooms have wireless presentation capability
            - Video conferencing: Zoom Rooms installed
            - Whiteboards and markers available
            - Power outlets and charging stations
            - Report equipment issues to facilities@company.com
            
            CATERING SERVICES:
            - Available for meetings >10 people
            - Order 24 hours in advance through admin team
            - Dietary restrictions accommodated
            - Cost center charging required
            
            WORKSPACE AMENITIES:
            - Open desk areas with height-adjustable desks
            - Quiet zones for focused work
            - Collaboration spaces with comfortable seating
            - Kitchen facilities: Coffee, tea, snacks, refrigerator
            - Printing/scanning stations on each floor
            
            OFFICE SUPPLIES:
            - Basic supplies available in supply closets
            - Request special items through admin team
            - Personal items: Reasonable use permitted
            - Ordering: Weekly batch orders to reduce costs
            
            VISITOR MANAGEMENT:
            - All visitors must be registered and escorted
            - Visitor badges required and must be visible
            - Guest WiFi available (password: CompanyGuest2024)
            - Parking: Visitor spots in front lot, validation available
            
            BUILDING ACCESS:
            - Key cards required for entry
            - Business hours: 6:00 AM - 10:00 PM weekdays
            - Weekend access: Request through security
            - Lost cards: Report immediately for deactivation/replacement
            """,
            "metadata": {"category": "Facilities", "department": "Administration", "last_updated": "2024-01-25"}
        }
    ]
    return documents

def run_advanced_demo():
    """Run the advanced RAG demo"""
    print("🚀 Advanced RAG Demo - Production Features")
    print("=" * 60)
    
    # Initialize advanced RAG system
    rag = AdvancedRAG()
    
    # Load comprehensive knowledge base
    print("📚 Loading comprehensive knowledge base...")
    documents = create_advanced_knowledge_base()
    
    total_chunks = 0
    for doc in documents:
        chunks = rag.add_document(doc["id"], doc["title"], doc["content"], doc["metadata"])
        total_chunks += chunks
    
    print(f"✅ Loaded {len(documents)} documents ({total_chunks} chunks)")
    
    # Display system statistics
    stats = rag.get_document_stats()
    print("📊 System Stats:")
    print(f"   • Documents: {stats['total_documents']}")
    print(f"   • Embedding Model: {stats['embedding_model']}")
    print(f"   • Embedding Dimension: {stats['embedding_dimension']}")
    
    # Test questions for demonstration
    test_questions = [
        "What is the vacation carryover policy?",
        "How much can I spend on hotels during business travel?",
        "What are the pricing tiers for CloudSync Pro?",
        "Can I work remotely 4 days a week?",
        "What are the password requirements?",
        "How do I book the board room for a client presentation?",
        "What's included with all product plans?",
        "Do I need manager approval for office supplies?",
        "What's the meal allowance for business travel?",
        "How do I report a security incident?"
    ]
    
    print(f"\n🧪 Testing with {len(test_questions)} advanced queries...")
    print("=" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n📝 Query {i}: {question}")
        print("-" * 40)
        
        # Process with advanced RAG
        response = rag.answer_question(question, top_k=3)
        
        print(f"💡 Answer: {response.answer[:200]}...")
        print(f"🎯 Confidence: {response.confidence:.3f}")
        print(f"⏱️  Processing Time: {response.processing_time:.3f}s")
        print(f"📚 Sources ({len(response.sources)}):")
        
        for source in response.sources[:2]:  # Show top 2 sources
            print(f"   • {source.document.title} (score: {source.similarity_score:.3f})")
    
    # Show query history
    print("\n📈 Recent Query History:")
    history = rag.get_query_history(5)
    for i, entry in enumerate(history[:3], 1):
        print(f"   {i}. {entry['query'][:50]}... (confidence: {entry['confidence']:.3f})")
    
    print("\n✅ Advanced RAG demo completed!")
    print("🔗 Features demonstrated:")
    print("   • Advanced embedding models")
    print("   • Vector similarity search")
    print("   • Document chunking and processing")
    print("   • Metadata storage and retrieval")
    print("   • Query history and analytics")
    print("   • Production-ready architecture")

if __name__ == "__main__":
    run_advanced_demo()
