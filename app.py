# 🎨 STREAMLIT MEDICAL RAG ASSISTANT WITH EVALUATION
import streamlit as st
import pandas as pd
import chromadb
import google.generativeai as genai
import os
from pathlib import Path
import numpy as np
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 Medical RAG Assistant")
st.markdown("### Diagnostic Reasoning for Clinical Notes")

# -------------------------------------------------------------------
# 1. FILE PATHS CONFIGURATION
# -------------------------------------------------------------------
# Define paths
BASE_DIR = Path(__file__).parent.absolute()
CHROMA_DB_FOLDER = BASE_DIR / "chromadb_clean"
GEMINI_API_KEY_FILE = BASE_DIR / "gemini_api_key.txt"
EVALUATION_RESULTS_FILE = BASE_DIR / "evaluation_results.csv"

# Debug: Show paths in expander for troubleshooting
with st.sidebar.expander("🔧 Debug Info"):
    st.write(f"**Base Dir:** {BASE_DIR}")
    st.write(f"**ChromaDB Path:** {CHROMA_DB_FOLDER}")
    st.write(f"**Exists:** {CHROMA_DB_FOLDER.exists()}")
    st.write(f"**API Key File:** {GEMINI_API_KEY_FILE}")
    st.write(f"**API File Exists:** {GEMINI_API_KEY_FILE.exists()}")

# -------------------------------------------------------------------
# 2. LOAD GEMINI API KEY
# -------------------------------------------------------------------
def load_gemini_api_key():
    """Load Gemini API key from file or environment"""
    api_key = None
    
    # 1. First try environment variable
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.sidebar.info("API key loaded from environment variable")
        return api_key
    
    # 2. Try to load from file
    if GEMINI_API_KEY_FILE.exists():
        try:
            with open(GEMINI_API_KEY_FILE, 'r') as f:
                api_key = f.read().strip()
                
            # Check if file is empty
            if not api_key:
                st.sidebar.warning("API key file exists but is empty")
                return None
                
            # Check if it looks like a valid API key (starts with AIza)
            if api_key.startswith("AIza"):
                st.sidebar.success("API key loaded from file")
                return api_key
            else:
                st.sidebar.warning(f"API key doesn't look valid (should start with 'AIza'): {api_key[:20]}...")
                return api_key
                
        except Exception as e:
            st.sidebar.error(f"Error reading API key file: {e}")
            return None
    
    # 3. Not found anywhere
    st.sidebar.warning("No API key found in file or environment")
    return None

# -------------------------------------------------------------------
# 3. INITIALIZE CHROMADB
# -------------------------------------------------------------------
@st.cache_resource
def load_chroma_db():
    """Load ChromaDB"""
    if not CHROMA_DB_FOLDER.exists():
        st.error(f"❌ ChromaDB folder not found at: {CHROMA_DB_FOLDER}")
        st.info("""
        Please ensure the 'chromadb_clean' folder contains:
        - ChromaDB data files
        - Should be in the same directory as this app
        """)
        return None
    
    try:
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_FOLDER))
        
        # Try to get collection - check for common collection names
        collection_names = chroma_client.list_collections()
        
        if not collection_names:
            st.error("❌ No collections found in ChromaDB")
            return None
        
        # Use the first collection found
        collection_name = collection_names[0].name
        collection = chroma_client.get_collection(name=collection_name)
        
        st.success(f"✅ Loaded ChromaDB collection: '{collection_name}'")
        return collection
        
    except Exception as e:
        st.error(f"❌ Error loading ChromaDB: {str(e)}")
        return None

# -------------------------------------------------------------------
# 4. SIDEBAR CONFIGURATION
# -------------------------------------------------------------------
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Key input section
    st.subheader("🔑 Gemini API Key Configuration")
    
    # Try to load API key automatically first
    gemini_api_key = load_gemini_api_key()
    
    # Show current API key status
    if gemini_api_key:
        st.success(f"✅ API key loaded: {gemini_api_key[:10]}...{gemini_api_key[-6:]}")
        api_key_method = "Use loaded key"
    else:
        st.warning("⚠️ No API key found")
        api_key_method = "Enter manually"
    
    # Let user choose method
    method = st.radio(
        "Choose input method:",
        ["Use loaded key", "Enter manually", "Load from file"],
        index=0 if gemini_api_key else 1
    )
    
    if method == "Enter manually":
        new_api_key = st.text_input("Enter your Gemini API key:", type="password")
        if new_api_key:
            gemini_api_key = new_api_key
            st.success("✅ Manual API key entered")
    
    elif method == "Load from file":
        if st.button("🔄 Reload from file"):
            gemini_api_key = load_gemini_api_key()
            if gemini_api_key:
                st.success("✅ API key reloaded from file")
                st.rerun()
    
    # Save API key to file
    if gemini_api_key and st.button("💾 Save API key to file", use_container_width=True):
        try:
            with open(GEMINI_API_KEY_FILE, 'w') as f:
                f.write(gemini_api_key.strip())
            st.success(f"✅ API key saved to: {GEMINI_API_KEY_FILE}")
        except Exception as e:
            st.error(f"Error saving API key: {e}")
    
    # Test API key button
    if gemini_api_key and st.button("🧪 Test API Connection", use_container_width=True):
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content("Say 'API connection successful' in one word.")
            st.success(f"✅ API Test: {response.text}")
        except Exception as e:
            st.error(f"❌ API Test Failed: {str(e)}")
    
    st.markdown("---")
    
    # Retrieval settings
    st.subheader("⚙️ Retrieval Settings")
    top_k = st.slider("Number of documents to retrieve", 1, 10, 3)
    
    # Disease filter
    st.subheader("🔍 Filter by Disease")
    diseases = ["All", "Acute Coronary Syndrome", "Diabetes", "Pneumonia", "COPD", "Hypertension"]
    selected_disease = st.selectbox("Select disease", diseases)
    
    st.markdown("---")
    
    # Load ChromaDB button
    st.subheader("📁 ChromaDB Status")
    if st.button("🔄 Load/Reload ChromaDB", use_container_width=True, type="primary"):
        st.cache_resource.clear()
        st.rerun()
    
    # Show current status
    collection = load_chroma_db()
    if collection:
        st.success(f"✅ ChromaDB loaded: {collection.count()} documents")
    else:
        st.error("❌ ChromaDB not loaded")
    
    st.markdown("---")
    
    # Example queries
    st.markdown("**📋 Example Queries:**")
    example_queries = [
        "chest pain treatment",
        "diabetes medication",
        "pneumonia symptoms",
        "hypertension management",
        "heart attack signs",
        "blood pressure control"
    ]
    
    for query in example_queries:
        if st.button(f"🔍 \"{query}\"", key=f"ex_{query}", use_container_width=True):
            st.session_state.query = query
            st.rerun()
    
    st.markdown("---")
    
    # RAG Evaluation Section
    st.subheader("📊 RAG Evaluation")
    
    if st.button("🧪 Run RAG Evaluation", use_container_width=True):
        st.session_state.run_evaluation = True
        st.rerun()
    
    st.markdown("---")
    
    # Required files info
    st.markdown("**📁 Required Files:**")
    st.code("""
📁 project_folder/
├── app.py              (this file)
├── chromadb_clean/     (ChromaDB storage)
├── gemini_api_key.txt  (store API key here)
└── requirements.txt    (dependencies)
    """)

# -------------------------------------------------------------------
# 5. INITIALIZE GEMINI
# -------------------------------------------------------------------
def initialize_gemini(api_key, model_name='gemini-2.5-flash'):
    """Initialize Gemini model"""
    if not api_key:
        st.error("❌ No Gemini API key provided")
        return None
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        return model
    except Exception as e:
        st.error(f"❌ Gemini initialization failed: {str(e)}")
        return None

# -------------------------------------------------------------------
# 6. OUT-OF-CONTEXT HANDLING
# -------------------------------------------------------------------
def handle_out_of_context(query, api_key):
    """Handle queries that are out of context using Gemini's general knowledge"""
    if not api_key:
        return "❌ Gemini API key required to answer general questions."
    
    try:
        model = initialize_gemini(api_key)
        if not model:
            return "❌ Failed to initialize Gemini."
        
        prompt = f"""You are a medical assistant. The user asked: "{query}"

This question is outside the scope of the available medical notes database.

Please provide a helpful, general medical answer based on your medical knowledge.
Be clear that this is general information, not based on specific patient notes.

Important: Always include a disclaimer that this is not medical advice and the user should consult a healthcare professional.

Answer:"""
        
        response = model.generate_content(prompt)
        return f"""**📝 General Medical Information (Not from specific patient notes):**

{response.text}

---
⚠️ **Disclaimer:** This is general medical information for educational purposes only. 
Always consult with a qualified healthcare professional for medical advice."""
    
    except Exception as e:
        return f"❌ Error generating general answer: {str(e)}"

# -------------------------------------------------------------------
# 7. RETRIEVAL FUNCTION
# -------------------------------------------------------------------
def retrieve_documents(query, top_k, disease_filter="All"):
    """Retrieve documents from ChromaDB"""
    try:
        if not collection:
            return []
        
        # Query ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=top_k * 3  # Get more to filter by disease
        )
        
        # Filter by disease if needed
        filtered_docs = []
        if results and results.get('documents'):
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                doc_disease = meta.get('disease', 'Unknown') if meta else 'Unknown'
                
                if disease_filter == "All" or doc_disease == disease_filter:
                    filtered_docs.append({
                        'text': doc,
                        'disease': doc_disease,
                        'patient_id': meta.get('patient_id', 'N/A') if meta else 'N/A',
                        'similarity': float(1 - dist) if isinstance(dist, (int, float)) else 0.0,
                        'word_count': len(str(doc).split())
                    })
        
        return filtered_docs[:top_k]
    
    except Exception as e:
        st.error(f"❌ Retrieval error: {e}")
        return []

# -------------------------------------------------------------------
# 8. GENERATION FUNCTION
# -------------------------------------------------------------------
def generate_answer(query, retrieved_docs, api_key, use_general_knowledge=False):
    """Generate answer using Gemini"""
    
    # If no documents retrieved and use_general_knowledge is True, use Gemini's general knowledge
    if not retrieved_docs and use_general_knowledge:
        return handle_out_of_context(query, api_key)
    
    if not retrieved_docs:
        return "No documents retrieved to generate answer."
    
    if not api_key:
        return "❌ Gemini API key required. Please configure it in the sidebar."
    
    # Initialize Gemini
    model = initialize_gemini(api_key)
    if model is None:
        return "❌ Gemini model failed to initialize. Check your API key."
    
    # Create context from retrieved documents
    context_parts = []
    for i, doc in enumerate(retrieved_docs):
        context_parts.append(
            f"--- DOCUMENT {i+1} ---\n"
            f"Disease: {doc['disease']}\n"
            f"Patient ID: {doc['patient_id']}\n"
            f"Relevance: {doc['similarity']:.2%}\n"
            f"Content: {doc['text'][:400]}..."
        )
    
    context = "\n\n".join(context_parts)
    
    # Create medical-focused prompt
    prompt = f"""# MEDICAL QUESTION ANSWERING TASK

## CONTEXT (Patient Medical Notes):
{context}

## QUESTION:
{query}

## INSTRUCTIONS:
1. Analyze the provided medical notes carefully
2. Extract relevant information that addresses the question
3. Provide a concise, evidence-based answer
4. Cite which document(s) support your answer
5. Note any limitations or missing information
6. If there's conflicting information, mention it

## ANSWER FORMAT:
Start with: "Based on the available medical notes..."

## ANSWER:"""
    
    try:
        with st.spinner("Generating answer with Gemini..."):
            response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Error generating answer: {str(e)}"

# -------------------------------------------------------------------
# 9. RAG EVALUATION FUNCTIONS
# -------------------------------------------------------------------
def evaluate_retrieval_performance(query_obj, top_k=3):
    """Evaluate retrieval performance for a query"""
    query = query_obj["query"]
    expected_diseases = set(query_obj["expected_diseases"])
    
    # Retrieve documents
    retrieved = retrieve_documents(query, top_k, "All")
    
    if not retrieved:
        return {
            "query": query,
            "retrieved_count": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "retrieved_diseases": [],
            "expected_diseases": list(expected_diseases)
        }
    
    # Calculate metrics
    retrieved_diseases = set([doc['disease'] for doc in retrieved])
    
    # Precision: % of retrieved docs that are relevant
    relevant_retrieved = len(retrieved_diseases.intersection(expected_diseases))
    precision = relevant_retrieved / len(retrieved) if retrieved else 0
    
    # Recall: % of expected diseases found
    recall = relevant_retrieved / len(expected_diseases) if expected_diseases else 0
    
    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "query": query,
        "retrieved_count": len(retrieved),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "retrieved_diseases": list(retrieved_diseases),
        "expected_diseases": list(expected_diseases)
    }

def evaluate_generation_performance(query_obj, retrieved_docs, api_key):
    """Evaluate answer generation quality"""
    if not retrieved_docs:
        return {
            "answer_length": 0,
            "contains_expected": 0,
            "coverage": 0,
            "coherence": 0
        }
    
    # Generate answer
    answer = generate_answer(query_obj["query"], retrieved_docs, api_key, use_general_knowledge=False)
    
    # Check if answer contains expected terms
    expected_terms = query_obj.get("expected_answer_contains", [])
    contains_count = 0
    if expected_terms:
        answer_lower = answer.lower()
        contains_count = sum(1 for term in expected_terms if term.lower() in answer_lower)
        coverage = contains_count / len(expected_terms)
    else:
        coverage = 0
    
    # Simple coherence score (0-1 based on length and structure)
    word_count = len(answer.split())
    coherence = min(1.0, word_count / 200)  # Normalize by 200 words
    
    return {
        "answer_length": word_count,
        "contains_expected": contains_count,
        "coverage": round(coverage, 3),
        "coherence": round(coherence, 3),
        "answer_preview": answer[:100] + "..." if len(answer) > 100 else answer
    }

def run_rag_evaluation():
    """Run comprehensive RAG evaluation"""
    
    if not collection:
        st.error("❌ ChromaDB must be loaded to run evaluation")
        return None
    
    if not gemini_api_key:
        st.error("❌ Gemini API key required for evaluation")
        return None
    
    # Test queries for evaluation
    test_queries = [
        {
            "query": "What treatments are used for chest pain?",
            "expected_diseases": ["Acute Coronary Syndrome", "Heart Failure"],
            "expected_answer_contains": ["aspirin", "nitroglycerin", "morphine", "treatment"]
        },
        {
            "query": "How is diabetes managed?",
            "expected_diseases": ["Diabetes"],
            "expected_answer_contains": ["insulin", "metformin", "diet", "glucose"]
        },
        {
            "query": "What are symptoms of pneumonia?",
            "expected_diseases": ["Pneumonia"],
            "expected_answer_contains": ["fever", "cough", "shortness", "breath"]
        }
    ]
    
    results = []
    with st.spinner("Running RAG evaluation..."):
        for i, query_obj in enumerate(test_queries):
            # Evaluate retrieval
            retrieval_result = evaluate_retrieval_performance(query_obj)
            
            # Get documents for generation evaluation
            retrieved = retrieve_documents(query_obj["query"], 3, "All")
            
            # Evaluate generation
            generation_result = evaluate_generation_performance(query_obj, retrieved, gemini_api_key)
            
            # Combine results
            combined = {**retrieval_result, **generation_result}
            results.append(combined)
    
    return results

# -------------------------------------------------------------------
# 10. MAIN INTERFACE
# -------------------------------------------------------------------
# Initialize Gemini
model = None

# Query input section
st.subheader("💭 Ask a Medical Question")

# Use session state to preserve query
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'run_evaluation' not in st.session_state:
    st.session_state.run_evaluation = False

query = st.text_input(
    "Enter your clinical question:",
    value=st.session_state.query,
    placeholder="e.g., What are common treatments for chest pain?",
    key="query_input"
)

# Action buttons
col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
with col1:
    retrieve_btn = st.button("🔍 Retrieve Documents", type="primary", disabled=collection is None, use_container_width=True)
with col2:
    generate_btn = st.button("🤖 Generate Answer", type="secondary", disabled=collection is None, use_container_width=True)
with col3:
    general_answer_btn = st.button("💡 General Knowledge", type="secondary", disabled=not gemini_api_key, use_container_width=True,
                                   help="Get answer using Gemini's general medical knowledge")

# Show warning if ChromaDB not loaded
if collection is None:
    st.warning("""
    ⚠️ **ChromaDB not loaded.** 
    
    Please ensure:
    1. 'chromadb_clean' folder is in the same directory as this app
    2. The folder contains valid ChromaDB files
    3. Click 'Load/Reload ChromaDB' in the sidebar
    """)

# Show warning if no API key
if not gemini_api_key:
    st.warning("""
    ⚠️ **Gemini API key not configured.**
    
    To generate answers:
    1. Enter your API key in the sidebar
    2. OR create a file called `gemini_api_key.txt` with your API key
    3. OR set environment variable: `GEMINI_API_KEY`
    
    Get API key from: https://makersuite.google.com/app/apikey
    """)

# -------------------------------------------------------------------
# 11. DISPLAY RESULTS
# -------------------------------------------------------------------
if retrieve_btn or generate_btn or general_answer_btn:
    if not query:
        st.warning("⚠️ Please enter a question first")
        st.stop()
    
    # Store query in session
    st.session_state.query = query
    
    # Handle General Knowledge button
    if general_answer_btn:
        if not gemini_api_key:
            st.error("❌ Gemini API key required for general knowledge answers")
        else:
            with st.spinner("💭 Consulting general medical knowledge..."):
                answer = handle_out_of_context(query, gemini_api_key)
            
            st.subheader("💡 General Medical Knowledge Answer")
            st.markdown("---")
            st.markdown(answer)
            st.markdown("---")
            st.info("ℹ️ This answer is based on general medical knowledge, not specific patient notes.")
    
    # Handle Retrieve and Generate buttons
    elif retrieve_btn or generate_btn:
        if not collection:
            st.error("❌ ChromaDB must be loaded first")
            st.stop()
        
        # Retrieve documents
        with st.spinner(f"🔍 Retrieving documents for: '{query}'..."):
            retrieved = retrieve_documents(query, top_k, selected_disease)
        
        # Display results section
        if retrieved:
            st.success(f"✅ Found {len(retrieved)} relevant document(s)")
            
            # Display retrieved documents
            st.subheader("📄 Retrieved Documents")
            
            for i, doc in enumerate(retrieved):
                with st.expander(
                    f"📋 Document {i+1}: {doc['disease']} (Relevance: {doc['similarity']:.1%})",
                    expanded=(i == 0)
                ):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.metric("Relevance", f"{doc['similarity']:.1%}")
                        st.metric("Word Count", doc['word_count'])
                        st.write(f"**Patient ID:** {doc['patient_id']}")
                        st.write(f"**Disease:** {doc['disease']}")
                    
                    with col_b:
                        st.text_area(
                            "Content:",
                            doc['text'],
                            height=200,
                            key=f"doc_text_{i}",
                            disabled=True
                        )
            
            # Generate answer if requested
            if generate_btn:
                if not gemini_api_key:
                    st.error("❌ Gemini API key required for generation")
                    st.info("Please enter your API key in the sidebar")
                else:
                    answer = generate_answer(query, retrieved, gemini_api_key)
                    
                    st.subheader("🤖 Generated Medical Answer")
                    st.markdown("---")
                    
                    # Display answer in a nice format
                    answer_container = st.container()
                    with answer_container:
                        st.markdown(answer)
                    
                    st.markdown("---")
                    
                    # Show metadata
                    with st.expander("📊 Generation Details"):
                        st.write(f"**Query:** {query}")
                        st.write(f"**Documents used:** {len(retrieved)}")
                        st.write(f"**Disease filter:** {selected_disease}")
                        
        else:
            # No documents found - offer general knowledge option
            st.warning("""
            ⚠️ **No relevant documents found in the database.**
            
            Options:
            1. Try different keywords
            2. Change disease filter to "All"
            3. Use one of the example queries
            4. Click 'General Knowledge' button for a general medical answer
            """)

# -------------------------------------------------------------------
# 12. RAG EVALUATION DISPLAY
# -------------------------------------------------------------------
if st.session_state.run_evaluation:
    st.session_state.run_evaluation = False
    
    st.subheader("📊 RAG System Evaluation")
    st.markdown("---")
    
    if not collection:
        st.error("❌ ChromaDB must be loaded to run evaluation")
    elif not gemini_api_key:
        st.error("❌ Gemini API key required for evaluation")
    else:
        results = run_rag_evaluation()
        
        if results:
            df_results = pd.DataFrame(results)
            
            # Display results in a nice format
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Average Precision", f"{df_results['precision'].mean():.3f}")
                st.metric("Average Recall", f"{df_results['recall'].mean():.3f}")
                st.metric("Average F1 Score", f"{df_results['f1'].mean():.3f}")
            
            with col2:
                st.metric("Average Coverage", f"{df_results['coverage'].mean():.3f}")
                st.metric("Average Coherence", f"{df_results['coherence'].mean():.3f}")
                st.metric("Total Queries", len(df_results))
            
            st.markdown("---")
            st.subheader("📋 Detailed Results")
            
            # Display table
            display_cols = ['query', 'retrieved_count', 'precision', 'recall', 'f1', 'coverage', 'coherence']
            st.dataframe(df_results[display_cols], use_container_width=True)
            
            # Show error analysis
            st.subheader("🔍 Error Analysis")
            
            issues = []
            for _, row in df_results.iterrows():
                if row['precision'] < 0.5:
                    issues.append(f"Low precision for: '{row['query']}'")
                if row['recall'] < 0.5:
                    issues.append(f"Low recall for: '{row['query']}'")
                if row['coverage'] < 0.5:
                    issues.append(f"Poor answer coverage for: '{row['query']}'")
            
            if issues:
                for issue in issues:
                    st.error(f"• {issue}")
            else:
                st.success("✅ No major issues found in evaluation")
            
            # Save results
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rag_evaluation_{timestamp}.csv"
                df_results.to_csv(BASE_DIR / filename, index=False)
                st.success(f"💾 Evaluation results saved to: {filename}")
            except Exception as e:
                st.warning(f"Could not save results: {e}")
        else:
            st.error("❌ Evaluation failed to produce results")

# -------------------------------------------------------------------
# 13. SYSTEM INFORMATION
# -------------------------------------------------------------------
with st.expander("ℹ️ System Information & Debug"):
    st.subheader("📊 Status")
    
    col_status1, col_status2 = st.columns(2)
    with col_status1:
        if collection:
            st.success("✅ ChromaDB: LOADED")
            st.write(f"**Documents:** {collection.count()}")
            st.write(f"**Collection:** {collection.name}")
        else:
            st.error("❌ ChromaDB: NOT LOADED")
    
    with col_status2:
        if gemini_api_key:
            st.success("✅ Gemini API: CONFIGURED")
            masked_key = f"{gemini_api_key[:8]}...{gemini_api_key[-6:]}"
            st.write(f"**Key:** {masked_key}")
        else:
            st.error("❌ Gemini API: NOT CONFIGURED")
            st.write("**Get API key:** https://makersuite.google.com/app/apikey")
    
    st.subheader("📁 File System")
    st.write(f"**Working Directory:** `{BASE_DIR}`")
    
    # List files in directory
    st.write("**Files in directory:**")
    files_list = []
    for item in BASE_DIR.iterdir():
        if item.is_dir():
            size = "DIR"
            icon = "📁"
        else:
            size_kb = item.stat().st_size / 1024
            size = f"{size_kb:.1f} KB"
            icon = "📄"
        
        files_list.append(f"{icon} {item.name} ({size})")
    
    for file_info in sorted(files_list):
        st.write(f"- {file_info}")
    
    # Show ChromaDB folder contents if exists
    if CHROMA_DB_FOLDER.exists():
        st.write(f"\n**Contents of '{CHROMA_DB_FOLDER.name}':**")
        try:
            chroma_items = list(CHROMA_DB_FOLDER.iterdir())
            if chroma_items:
                for item in chroma_items:
                    if item.is_dir():
                        st.write(f"  - 📁 {item.name}/")
                    else:
                        size_kb = item.stat().st_size / 1024
                        st.write(f"  - 📄 {item.name} ({size_kb:.1f} KB)")
            else:
                st.write("  *(empty)*")
        except:
            st.write("  *(cannot list contents)*")
    
    # Show API key file contents
    if GEMINI_API_KEY_FILE.exists():
        st.write(f"\n**Contents of '{GEMINI_API_KEY_FILE.name}':**")
        try:
            with open(GEMINI_API_KEY_FILE, 'r') as f:
                content = f.read().strip()
                if content:
                    st.code(content[:50] + "..." if len(content) > 50 else content)
                else:
                    st.write("*(empty file)*")
        except:
            st.write("*(cannot read file)*")

# -------------------------------------------------------------------
# 14. FOOTER
# -------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
    <h4 style='color: #ff4b4b;'>⚠️ IMPORTANT DISCLAIMER</h4>
    <p style='color: #666;'>
    <strong>FOR EDUCATIONAL AND RESEARCH PURPOSES ONLY</strong><br>
    This tool is not for clinical decision-making.<br>
    Always consult qualified healthcare professionals for medical advice.<br>
    Do not use for diagnosis or treatment decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# 15. CREATE API KEY FILE IF NOT EXISTS
# -------------------------------------------------------------------
if not GEMINI_API_KEY_FILE.exists():
    st.sidebar.info("ℹ️ API key file will be created when you save a key")