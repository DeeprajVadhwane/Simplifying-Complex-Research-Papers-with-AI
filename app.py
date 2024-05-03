import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import NLTKTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader
from langchain_community.document_loaders import OnlinePDFLoader
# API Key
google_api_key ="AIzaSyCygnvCyqn3FjNTO_MIRLIKUhw7S5W-ulk"

# Title and description
st.title("Research Paper Reader & QA")
st.write("Upload a PDF, provide a website link, or input custom text. Then ask questions about the content.")

# Function to load and split documents into text chunks
def load_and_split(docs):
    text_splitter = NLTKTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# Function to create a retriever from document chunks
def get_vector_store(text_chunks):
    embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=google_api_key, model="models/embedding-001")
    db = Chroma.from_documents(text_chunks, embedding_model)
    retriever = db.as_retriever(search_kwargs={"k": 10})
    return retriever

# Function to create a retrieval-augmented generation (RAG) chain
def get_rag_chain(retriever):
    chat_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("""You are a helpful AI assistant with the ability to analyze various data types, including PDFs, text, and web links. 
Your task is to understand the provided data, extract key information, and respond in a clear, structured manner with bullet points or headings. 
If the data contains images, mathematical equations, or technical terms, explain them in simple language.
Provide the output in a structured and easy-to-read format.""")
    ])
    chat_model = ChatGoogleGenerativeAI(google_api_key=google_api_key, model="gemini-1.5-pro-latest")
    output_parser = StrOutputParser()

    def format_docs(docs):
        if isinstance(docs, list) and all(hasattr(doc, 'page_content')):
            return "\n\n".join(doc.page_content for doc in docs)
        else:
            return "Error: Invalid content format."

    rag_chain = {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    } | chat_template | chat_model | output_parser

    return rag_chain

# Streamlit app to choose input type
st.header("Data Input")
input_type = st.radio("Select input type:", ["PDF", "Website", "Custom Text"])

# Load documents based on input type
docs = []
if input_type == "PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded_file.read())
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()

elif input_type == "Website":
    web_link = st.text_input("Enter a website link")
    if web_link:
        loader = PyPDFLoader(web_link)
        docs = loader.load()

elif input_type == "Custom Text":
    custom_text = st.text_area("Enter custom text")
    if custom_text:
        docs.append({"page_content": custom_text})

# Display the loaded content
st.write("Loaded Content")
if docs:
    content = "\n\n".join(
        doc["page_content"] if isinstance(doc, dict) else doc.page_content
        for doc in docs
    )
    if not content.strip():
        st.warning("The loaded content is empty or unreadable. Please check your input.")
    else:
        st.text(content[:2000])  
        if st.button("Show More"):
            st.text(content[500:])

# Create RAG chain and ask questions
if docs and st.checkbox("Would you like to ask a question about the loaded content?"):
    chunks = load_and_split(docs)
    retriever = get_vector_store(chunks)
    rag_chain = get_rag_chain(retriever)

    st.header("Question and Answer")
    question = st.text_area("Ask a question")
    if st.button("Submit Question"):
        try:
            response = rag_chain.invoke(question)
            st.write("Response:")
            st.markdown(response)  # Use markdown for better readability
        except Exception as e:
            st.error(f"Error generating response: {e}")
