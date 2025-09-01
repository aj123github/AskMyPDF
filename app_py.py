import streamlit as st
import tempfile
import os

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader

# ✅ Updated imports (new structure)
from langchain_openai import OpenAIEmbeddings, OpenAI


# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="AskMyPDF", page_icon="📘")
st.title("📘 AskMyPDF: Interactive Q&A System using RAG")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Build retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0),
        retriever=retriever,
        return_source_documents=True
    )

    # User Question
    query = st.text_input("💬 Ask a question about the PDF:")

    if query:
        with st.spinner("Generating answer..."):
            result = qa_chain({"query": query})
            answer = result["result"]
            sources = result["source_documents"]

        st.subheader("📌 Answer")
        st.write(answer)

        # Show sources (optional)
        with st.expander("Show retrieved context"):
            for i, doc in enumerate(sources):
                st.markdown(f"**Chunk {i+1}:** {doc.page_content}")

    # Cleanup
    os.remove(tmp_path)
