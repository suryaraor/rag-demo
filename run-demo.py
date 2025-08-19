#!/usr/bin/env python3
"""
RAG Demo Launcher
Easy way to run different RAG demonstrations.
"""

import sys
import os
from pathlib import Path

def print_banner():
    """Print welcome banner"""
    print("🧠" + "=" * 58 + "🧠")
    print("     RAG Demo - Understanding Retrieval-Augmented Generation")
    print("🧠" + "=" * 58 + "🧠")
    print()

def check_requirements():
    """Check if required packages are available"""
    missing_packages = []
    
    try:
        import numpy
    except ImportError:
        missing_packages.append("numpy")
    
    try:
        import sentence_transformers
        print("✅ sentence-transformers available (advanced features enabled)")
    except ImportError:
        print("⚠️  sentence-transformers not found (using simple embeddings)")
    
    try:
        import faiss
        print("✅ FAISS available (efficient vector search enabled)")
    except ImportError:
        print("⚠️  FAISS not found (using numpy search)")
    
    if missing_packages:
        print(f"\n❌ Missing required packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    return True

def show_menu():
    """Display demo options"""
    print("\n🎯 Choose your RAG demo experience:\n")
    print("1. 🌐 Interactive Web Demo (No setup required)")
    print("   └── Open index.html in your browser")
    print("   └── Visual explanation with interactive examples")
    print()
    print("2. 🐍 Simple Python Demo")
    print("   └── Basic RAG implementation with explanations")
    print("   └── Interactive Q&A with sample knowledge base")
    print()
    print("3. 🚀 Advanced Python Demo")
    print("   └── Production-ready features and vector databases")
    print("   └── Comprehensive testing with real embeddings")
    print()
    print("4. 📁 Browse Sample Data")
    print("   └── View the knowledge base documents")
    print()
    print("5. ❓ Help - Understanding RAG")
    print("   └── Detailed explanation of RAG concepts")
    print()
    print("0. 🚪 Exit")

def run_web_demo():
    """Instructions for web demo"""
    html_path = Path("index.html")
    if html_path.exists():
        print("🌐 Web Demo Instructions:")
        print(f"1. Open {html_path.absolute()} in your web browser")
        print("2. Navigate through the tabs to learn RAG concepts")
        print("3. Try the interactive demo with sample questions")
        print("\n💡 The web demo provides visual explanations and simulated RAG responses")
    else:
        print("❌ index.html not found in current directory")

def run_simple_demo():
    """Run simple Python demo"""
    try:
        print("🚀 Starting Simple RAG Demo...")
        import subprocess
        subprocess.run([sys.executable, "simple-rag-demo.py"], check=True)
    except FileNotFoundError:
        print("❌ simple-rag-demo.py not found")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running simple demo: {e}")

def run_advanced_demo():
    """Run advanced Python demo"""
    try:
        print("🚀 Starting Advanced RAG Demo...")
        import subprocess
        subprocess.run([sys.executable, "advanced-rag-demo.py"], check=True)
    except FileNotFoundError:
        print("❌ advanced-rag-demo.py not found")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running advanced demo: {e}")

def browse_sample_data():
    """Show sample data files"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found")
        return
    
    print("📁 Sample Knowledge Base Documents:\n")
    
    for file_path in data_dir.glob("*.txt"):
        print(f"📄 {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Show first few lines
                lines = content.split('\n')[:10]
                for line in lines:
                    if line.strip():
                        print(f"   {line[:80]}...")
                        break
        except Exception as e:
            print(f"   Error reading file: {e}")
        print()

def show_help():
    """Show detailed RAG explanation"""
    print("❓ Understanding RAG (Retrieval-Augmented Generation)\n")
    
    print("🎯 What is RAG?")
    print("RAG combines information retrieval with text generation to provide")
    print("accurate, contextual responses based on your specific knowledge base.\n")
    
    print("🔄 How RAG Works:")
    print("1. 📚 Document Ingestion: Load and process your documents")
    print("2. ✂️  Text Chunking: Split documents into searchable pieces")
    print("3. 🧮 Embedding Generation: Convert text to vector representations")
    print("4. 🗄️  Vector Storage: Store embeddings in searchable database")
    print("5. 🔍 Query Processing: Convert user question to vector")
    print("6. 📊 Similarity Search: Find most relevant document chunks")
    print("7. 🔗 Context Injection: Combine retrieved info with question")
    print("8. 🤖 Response Generation: Generate answer using LLM + context\n")
    
    print("✅ Benefits of RAG:")
    print("• Access to current and private information")
    print("• Reduced hallucinations in AI responses")
    print("• Source attribution and transparency")
    print("• No need to retrain AI models")
    print("• Cost-effective compared to fine-tuning\n")
    
    print("🛠️  This Demo Includes:")
    print("• Simple implementation (educational)")
    print("• Advanced implementation (production-ready)")
    print("• Interactive web interface")
    print("• Sample knowledge base documents")
    print("• Different embedding and search strategies\n")

def main():
    """Main demo launcher"""
    print_banner()
    
    if not check_requirements():
        print("\n💡 You can still run the web demo without additional packages!")
        print("   Open index.html in your browser to get started.")
        return
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🔢 Enter your choice (0-5): ").strip()
            
            if choice == "0":
                print("\n👋 Thanks for exploring RAG! Happy building!")
                break
            elif choice == "1":
                run_web_demo()
            elif choice == "2":
                run_simple_demo()
            elif choice == "3":
                run_advanced_demo()
            elif choice == "4":
                browse_sample_data()
            elif choice == "5":
                show_help()
            else:
                print("❌ Invalid choice. Please enter 0-5.")
            
            if choice != "0":
                input("\n⏸️  Press Enter to continue...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for exploring RAG! Happy building!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
