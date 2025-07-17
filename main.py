import os
import streamlit as st
import pickle
import time
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain import OpenAI
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_huggingface.pkl"

main_placeholder = st.empty()
llm=OpenAI(openai_api_key="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",temperature=0, max_tokens=500)
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # create embeddings using HuggingFace and save it to FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_hf = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_hf, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
          
            chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff",
                retriever=vectorstore.as_retriever(),
            )
            result = chain.run(query)

            st.header("Answer")
            st.write(result)
