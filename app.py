import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

st.set_page_config(page_title="Hackathon Copilot", page_icon="⚡")
st.title("Open-Weight RAG Copilot")
st.write("PDF research assistant powered by Llama 3 via Groq.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_file = st.file_uploader("Drop a PDF here", type="pdf")
    
    if st.button("Process Document") and uploaded_file is not None:
        with st.spinner("Embedding via HuggingFace..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # Uses open-source embeddings directly in Python
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            st.success("Document embedded! You can now chat.")
            os.remove(tmp_file_path)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_input := st.chat_input("Ask a question about the uploaded document..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if "retriever" in st.session_state:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                
                # 1. MANUAL RAG: Retrieve the text chunks bypassing LangChain chains
                retrieved_docs = st.session_state.retriever.invoke(user_input)
                context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
                
                # 2. Inject the chunks directly into the prompt string
                final_prompt = f"""You are an expert AI coding and research assistant.
Use the following retrieved pieces of context to answer the user's question. 
If you don't know the answer, just say that you don't know.

Context:
{context_text}

Question: {user_input}

Helpful Answer:"""

                # 3. Pass the string directly to Groq
                llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
                response = llm.invoke(final_prompt)
                answer = response.content
                
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.warning("Please upload and process a PDF document first.")