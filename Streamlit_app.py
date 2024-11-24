import streamlit as st
import pandas as pd
import pydeck as pdk
import json
from pdf_reader import invoice_reader_pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


def intro():

    st.write("# Welcome to Streamlit! 👋")
    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        Streamlit is an open-source app framework built specifically for
        Machine Learning and Data Science projects.

        **👈 Select a demo from the dropdown on the left** to see some examples
        of what Streamlit can do!

        ### Want to learn more?

        - Check out [streamlit.io](https://streamlit.io)
        - Jump into our [documentation](https://docs.streamlit.io)
        - Ask a question in our [community
          forums](https://discuss.streamlit.io)

        ### See more complex demos

        - Use a neural net to [analyze the Udacity Self-driving Car Image
          Dataset](https://github.com/streamlit/demo-self-driving)
        - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
    )

def invoice_reader():

    files: list = []
    document: str = ''
    config: json = read_config()
    with st.sidebar:
        # st.sidebar.success("This is Invoice Reader Application")
        st.info(body="This is Invoice reader application. Please upload a file to get results.", icon=":material/description:")
        files = st.file_uploader(label="Upload files", type="pdf", accept_multiple_files=True)
        user_question = st.sidebar.text_input("Type Your question here")
    
    if len(files) != 0:
        for file in files:
            document += invoice_reader_pypdf(file=file)

        text_splitter = RecursiveCharacterTextSplitter(
            separators="\n",
            chunk_size=1000,
            chunk_overlap=150,
            length_function=len
        )
        chunks = text_splitter.split_text(document)

        # generating embedding
        embeddings = OpenAIEmbeddings(openai_api_key=config["open_ai_api_key"])

        # creating vector store - FAISS
        vector_store = FAISS.from_texts(chunks, embeddings)

        # get user question
        user_question = st.sidebar.text_input("Type Your question here")

        # do similarity search
        if user_question:
            match = vector_store.similarity_search(user_question)
            #st.write(match)


            #define the LLM
            llm = ChatOpenAI(
                openai_api_key = config["open_ai_api_key"],
                temperature = 0,
                max_tokens = 1000,
                model_name = "babbage-002"
            )


            #output results
            #chain -> take the question, get relevant document, pass it to the LLM, generate the output
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = match, question = user_question)
            st.write(response)

def mapping_demo():


    from urllib.error import URLError

    st.markdown(f"# {list(page_names_to_funcs.keys())[2]}")
    st.write(
        """
        This demo shows how to use
[`st.pydeck_chart`](https://docs.streamlit.io/develop/api-reference/charts/st.pydeck_chart)
to display geospatial data.
"""
    )

def read_config():
    with open ('C:/Users/Programming/Documents/Python/AI/Invoice_model/env.json', 'r') as file:
        config = json.load(file)
    return config



page_names_to_funcs = {
    "Information": intro,
    "Invoice Reader": invoice_reader
}

demo_name = st.sidebar.selectbox("Choose an application", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()