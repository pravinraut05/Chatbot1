import streamlit as st
import os
import tempfile
import requests
import zipfile
import io
import shutil
import nltk
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader

# Fix the import issues
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # Updated import
    from langchain_community.vectorstores import FAISS  # Updated import
    from langchain.chains import ConversationalRetrievalChain
    from langchain.document_loaders import (
        PyPDFLoader,
        TextLoader,
        Docx2txtLoader
    )
    from langchain.schema import Document
    from langchain.memory import ConversationBufferMemory
    from langchain_experimental.agents import create_pandas_dataframe_agent  # Fixed import
    from langchain.agents.agent_types import AgentType
    import glob
    import json
    from typing import List, Dict, Any
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    st.error(f"LangChain import error: {e}")
    st.error("Please install the required packages: pip install langchain langchain-openai langchain-community langchain-experimental")
    LANGCHAIN_AVAILABLE = False

# Set NLTK_DATA path to a writable directory
try:
    nltk_data_dir = os.path.join(tempfile.gettempdir(), "nltk_data")
    os.makedirs(nltk_data_dir, exist_ok=True)
    nltk.data.path.append(nltk_data_dir)
    os.environ["NLTK_DATA"] = nltk_data_dir
    nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_dir, quiet=True)
    NLTK_READY = True
except Exception as e:
    st.warning(f"NLTK setup issue: {e}")
    NLTK_READY = False

# Page configuration
st.set_page_config(
    page_title="Kaizen Engineers Knowledge Base",
    page_icon="📚",
    layout="wide"
)

# Check if all dependencies are available
if not LANGCHAIN_AVAILABLE:
    st.error("❌ Required dependencies are missing. Please install them first.")
    st.code("pip install langchain langchain-openai langchain-community langchain-experimental streamlit pandas numpy requests PyPDF2 python-docx nltk faiss-cpu")
    st.stop()

# GitHub ZIP URL
GITHUB_ZIP_URL = "https://github.com/pravinraut05/Dataset/raw/main/orders.zip"

# Initialize session state variables with error handling
def initialize_session_state():
    """Initialize session state variables safely"""
    defaults = {
        "conversation": None,
        "csv_agent": None,
        "dataframes": {},
        "chat_history": [],
        "processed_data": False,
        "temp_dir": None,
        "initialization_started": False,
        "data_summary": {},
        "initialization_complete": False,
        "initialization_error": None
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

# Initialize session state
initialize_session_state()

def download_and_extract_zip(github_zip_url):
    """Download and extract ZIP file with better error handling"""
    try:
        st.info("📥 Downloading data from GitHub...")
        temp_dir = tempfile.mkdtemp()
        
        response = requests.get(github_zip_url, timeout=30)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.success("✅ Data downloaded successfully!")
        return temp_dir
        
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error downloading data: {e}")
        return None
    except zipfile.BadZipFile:
        st.error("❌ Downloaded file is not a valid ZIP archive")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error downloading data: {e}")
        return None

def load_csv_files(directory_path):
    """Load all CSV files and return as pandas DataFrames"""
    dataframes = {}
    csv_documents = []
    
    csv_files = glob.glob(f"{directory_path}/**/*.csv", recursive=True)
    
    if not csv_files:
        st.warning("No CSV files found in the dataset")
        return dataframes, csv_documents
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, csv_path in enumerate(csv_files):
        try:
            filename = os.path.basename(csv_path)
            status_text.text(f"Loading {filename}...")
            
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None and not df.empty:
                # Clean the dataframe
                df = df.dropna(how='all')
                df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                
                # Store dataframe
                dataframes[filename] = df
                
                # Create summary document for vector search
                num_rows, num_cols = df.shape
                columns = df.columns.tolist()
                
                summary_content = [
                    f"CSV File: {filename}",
                    f"Description: This dataset contains {num_rows} rows and {num_cols} columns",
                    f"Columns: {', '.join(columns)}",
                    f"Data types: {', '.join([f'{col}: {str(df[col].dtype)}' for col in columns[:10]])}"
                ]
                
                # Add sample data for context
                if num_rows > 0:
                    summary_content.append("Sample data preview:")
                    sample_data = df.head(3).to_string(index=False)
                    summary_content.append(sample_data)
                
                # Add statistical info for numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    summary_content.append(f"Numeric columns with statistics: {', '.join(numeric_cols)}")
                    for col in numeric_cols[:5]:
                        stats = df[col].describe()
                        summary_content.append(f"{col}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")
                
                # Create document for vector search
                doc = Document(
                    page_content="\n".join(summary_content),
                    metadata={
                        "source": csv_path,
                        "filename": filename,
                        "type": "csv_summary",
                        "rows": num_rows,
                        "columns": num_cols
                    }
                )
                csv_documents.append(doc)
                
        except Exception as e:
            st.warning(f"⚠️ Could not load {filename}: {str(e)}")
            continue
        
        # Update progress
        progress_bar.progress((i + 1) / len(csv_files))
    
    progress_bar.empty()
    status_text.empty()
    
    return dataframes, csv_documents

# Rest of your functions remain the same, but add this initialization check:

def safe_initialize_knowledge_base(openai_api_key):
    """Safely initialize knowledge base with comprehensive error handling"""
    try:
        st.session_state.initialization_error = None
        
        # Validate API key
        if not openai_api_key or len(openai_api_key) < 20:
            raise ValueError("Invalid OpenAI API key")
        
        # Test API key
        try:
            test_llm = ChatOpenAI(
                temperature=0,
                model_name="gpt-3.5-turbo",
                openai_api_key=openai_api_key,
                max_tokens=10
            )
            # Make a simple test call
            test_response = test_llm.invoke("Hello")
        except Exception as e:
            raise ValueError(f"OpenAI API key validation failed: {str(e)}")
        
        # Download and extract data
        temp_dir = download_and_extract_zip(GITHUB_ZIP_URL)
        
        if not temp_dir:
            raise Exception("Failed to download and extract data")
        
        st.session_state.temp_dir = temp_dir
        
        # Load CSV files
        dataframes, csv_documents = load_csv_files(temp_dir)
        st.session_state.dataframes = dataframes
        
        # Continue with rest of initialization...
        # (Include your other initialization code here)
        
        if dataframes:
            st.session_state.processed_data = True
            st.session_state.initialization_complete = True
            st.success(f"✅ Successfully loaded {len(dataframes)} CSV files!")
        else:
            st.warning("⚠️ No data files could be loaded")
            
    except Exception as e:
        st.session_state.initialization_error = str(e)
        st.error(f"❌ Initialization failed: {str(e)}")
        
        # Cleanup on error
        if st.session_state.temp_dir and os.path.exists(st.session_state.temp_dir):
            try:
                shutil.rmtree(st.session_state.temp_dir)
            except:
                pass
        st.session_state.temp_dir = None

# Main app
st.title("📚 Kaizen Engineers Knowledge Base")

# API Key handling with better validation
openai_api_key = os.environ.get("OPENAI_API_KEY", "")

if not openai_api_key:
    st.warning("🔑 Please enter your OpenAI API key to continue")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Your OpenAI API key (starts with 'sk-')")
    
    if openai_api_key:
        if not openai_api_key.startswith('sk-'):
            st.error("❌ Invalid API key format. OpenAI API keys start with 'sk-'")
        else:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ API key accepted!")

# Initialize knowledge base with better error handling
if openai_api_key and not st.session_state.initialization_started and not st.session_state.initialization_complete:
    st.session_state.initialization_started = True
    
    with st.spinner("🚀 Initializing knowledge base..."):
        safe_initialize_knowledge_base(openai_api_key)

# Show initialization status
if st.session_state.initialization_error:
    st.error(f"❌ Initialization Error: {st.session_state.initialization_error}")
    if st.button("🔄 Retry Initialization"):
        st.session_state.initialization_started = False
        st.session_state.initialization_error = None
        st.rerun()

elif st.session_state.processed_data:
    st.success("✅ Knowledge base ready!")
    
    # Your main chat interface code here...
    st.header("💬 Chat with Your Data")
    
    user_question = st.chat_input("Ask a question about your data...")
    if user_question:
        st.write(f"You asked: {user_question}")
        # Add your chat handling logic here

else:
    if openai_api_key and st.session_state.initialization_started:
        st.info("⏳ Setting up knowledge base...")
    else:
        st.info("👆 Enter your OpenAI API key above to get started")
