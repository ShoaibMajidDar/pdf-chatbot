import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ['OPENAI_API_KEY'] = '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'

uploaded_file = st.file_uploader('Choose your .pdf file', type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)


    emebeddings = OpenAIEmbeddings()

    vectorstore = FAISS.from_texts(chunks, embedding = emebeddings)

    prompt = st.text_input('ask any question')

    if prompt:
        docs = vectorstore.similarity_search(query=prompt, k=3)

        llm = OpenAI(temperature = 0.8)
        chain = load_qa_chain(llm=llm, chain_type='stuff', verbose=True)
        response = chain.run(input_documents=docs, question=prompt)

        if response != " I don't know.":
            st.write(response)
        else:
            response = llm(prompt)
            st.write(response)