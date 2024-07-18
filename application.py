from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

load_dotenv()

# Initialize the model and embedding
model = ChatGroq(
    temperature=0,
    model="llama3-70b-8192",
    api_key=os.environ.get('GROQ_API_KEY'),
)

embedding = HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-base-en',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Set the page title
st.title("Talk to Your PDF")

# File uploader for the PDF
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

# Directory to save uploaded PDFs
upload_directory = "uploaded_pdfs"
os.makedirs(upload_directory, exist_ok=True)

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

def chat(query_text, vectordb):
    results = vectordb.similarity_search(query_text, k=3)
    context_text = "\n\n - -\n\n".join([doc.page_content for doc in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    response_text = model.predict(prompt)

    return response_text

def create_documents(path):
    loader = PyPDFLoader(file_path=path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(documents)
    return chunked_documents

if uploaded_file:
    # Save the uploaded PDF locally
    file_path = os.path.join(upload_directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF file uploaded successfully!")

    st.write("Creating documents....")
    documents = create_documents(file_path)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding
    )

    # Text input for user's question
    user_question = st.text_input("Ask a question about the PDF")

    if user_question:
        # Button to submit the question
            st.write("Processing your question...")

            # Generate and display the answer
            answer = chat(user_question, vectordb)
            st.write(f"**Answer:** {answer}")

else:
    st.info("Please upload a PDF file to get started.")
