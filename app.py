import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA, AnalyzeDocumentChain
from langchain.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile
import pysqlite3
import sys
import requests

# Fix the sqlite3 module issue
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Page configuration
st.set_page_config(page_title='🦜🔗 Advanced Doc and Code App', layout="wide")

# Initialize session state variables
if 'document_list' not in st.session_state:
    st.session_state['document_list'] = []
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []
if 'api_key' not in st.session_state:
    st.session_state['api_key'] = ''
if 'together_api_key' not in st.session_state:
    st.session_state['together_api_key'] = ''

def add_to_sidebar(doc_name):
    """Update the sidebar with the new document."""
    if doc_name not in st.session_state['document_list']:
        st.session_state['document_list'].append(doc_name)

def load_document(file=None, url=None):
    """Load a document from a file or URL."""
    if file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()
        add_to_sidebar(file.name)
        return documents
    elif url is not None:
        loader = WebBaseLoader(url)
        documents = loader.load()
        add_to_sidebar(url)
        return documents

def generate_response(documents, openai_api_key, query_text):
    """Generate a response from the loaded documents."""
    try:
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings, persist_directory="chromadb_storage")
        db.persist()
        retriever = db.as_retriever(search_kwargs={"k": 3})  # Fetch top 3 relevant chunks
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(
                openai_api_key=openai_api_key,
                max_tokens=150  # Limit response length
            ),
            chain_type='stuff',
            retriever=retriever
        )
        return qa.run(query_text)
    except Exception as e:
        return f"Error: {str(e)}"

def generate_code(prompt, together_api_key):
    """Generate code using Together.ai and Code Llama."""
    try:
        url = "https://api.together.ai/code"
        headers = {"Authorization": f"Bearer {together_api_key}"}
        data = {"prompt": prompt, "model": "code-llama"}
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get('code', "No code returned")
        else:
            return f"Error: {response.json().get('message', 'Unknown error')}"
    except Exception as e:
        return f"Error: {str(e)}"

def summarize_document(documents):
    """Summarize the loaded documents."""
    summarizer = AnalyzeDocumentChain()
    return summarizer.run(documents)

# Sidebar
with st.sidebar:
    st.image("https://lwfiles.mycourse.app/65a58160c1646a4dce257fac-public/a82c64f84b9bb42db4e72d0d673a50d0.png", use_column_width=True)
    st.session_state['api_key'] = st.text_input("OpenAI API Key", type="password", placeholder="Enter your OpenAI API key")
    st.session_state['together_api_key'] = st.text_input("Together.ai API Key", type="password", placeholder="Enter your Together.ai API key")
    st.write("**Loaded Documents**")
    for doc in st.session_state['document_list']:
        st.write(f"- {doc}")

# Tabbed layout
tabs = st.tabs(["Document Q&A", "Code Generation", "Document Summarization", "Sentiment Analysis", "Download Document"])

# Tab 1: Document Q&A
with tabs[0]:
    st.title("Document Question & Answer")
    uploaded_file = st.file_uploader('Upload a PDF document', type='pdf')
    uploaded_url = st.text_input('Enter a website URL (optional)')
    documents = []
    if uploaded_file:
        documents = load_document(file=uploaded_file)
    elif uploaded_url:
        documents = load_document(url=uploaded_url)
    query_text = st.text_input('Enter your question:', placeholder='Ask something about the loaded documents.', disabled=not documents)
    if st.session_state['api_key'] and query_text and documents:
        with st.spinner('Generating response...'):
            response = generate_response(documents, st.session_state['api_key'], query_text)
            st.session_state['query_history'].append((query_text, response))
            st.write("**Response:**", response)

# Tab 2: Code Generation
with tabs[1]:
    st.title("Code Generation with Together.ai and Code Llama")
    code_prompt = st.text_area("Enter your coding prompt:")
    if st.session_state['together_api_key'] and code_prompt:
        with st.spinner("Generating code..."):
            generated_code = generate_code(code_prompt, st.session_state['together_api_key'])
            st.code(generated_code, language="python")

# Tab 3: Document Summarization
with tabs[2]:
    st.title("Summarize Documents")
    if documents:
        if st.button("Summarize Document"):
            summary = summarize_document(documents)
            st.write("**Summary:**", summary)

# Tab 4: Sentiment Analysis
with tabs[3]:
    st.title("Sentiment Analysis")
    if documents:
        sentiment_button = st.button("Analyze Sentiment")
        if sentiment_button:
            sentiment = "Positive"  # Placeholder for sentiment analysis logic
            st.write("**Sentiment:**", sentiment)

# Tab 5: Download Document
with tabs[4]:
    st.title("Download Loaded Documents")
    if documents:
        download_button = st.download_button(
            label="Download Document",
            data="\n".join([doc.page_content for doc in documents]),
            file_name="document.txt",
            mime="text/plain"
        )
