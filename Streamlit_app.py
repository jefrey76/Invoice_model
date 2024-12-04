import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import json
from pdf_reader import *
from answer_converter import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate


class Templates:

    templateForDocs = """Answer the question based on the provided invoices. Try to avoid adding introduction and explaining the answer, focus only on exact answer.
        If you will be asked about the table format, provide only table.

        Invoices: {documents}

        Question: {query}

        Answer: """

    templateForChat = """Answer the question below.

        Here is the conversation history: {context}.
        If you can't find the answer based on the history, use your data to provide the details.

        Question: {question}

        Answer
        """

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

def get_prompt_template(templateName: str)->PromptTemplate:

    if templateName == "docs":
        template = Templates.templateForDocs
    elif templateName == "chat":
        template = Templates.templateForChat
    
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template

def get_llm()->Ollama:

    config: json = read_config()
    ollama = Ollama(
        base_url=config['local_host'], 
        model=config['model'],
        temperature=config['temperature']
        )
    return ollama

def get_splitter()->RecursiveCharacterTextSplitter:
    
    config: json = read_config()
    text_splitter = RecursiveCharacterTextSplitter(
        separators=config['separators'],
        chunk_size=config['chunk_size'],
        chunk_overlap=config['chunk_overlap'],
        length_function=len
    )
    return text_splitter

def get_data_transformer_function(func_key: str, ai_response: str):

    function_dict = {
        "table": prepare_table,
        "old_invoice": get_oldest_invoice,
        "all_items": get_all_items,
        "chart": prepare_bar_chart
        }
    function_dict[func_key](ai_response)

def invoice_reader_llama_manual():

    files: list = []
    document: str = ''
    with st.sidebar:
        st.info(body="This is Invoice reader application. Please upload a file to get results.", icon=":material/description:")
        files = st.file_uploader(label="Upload files", type="pdf", accept_multiple_files=True)
    
    if len(files) != 0:
        for file in files:
            document += invoice_reader_pypdf(file=file)

        text_splitter = get_splitter()
        chunks = text_splitter.split_text(document)
        embeddings=GPT4AllEmbeddings()
        # embeddings = OllamaEmbeddings(model="llama3")
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

        user_question = st.text_input("Type your question here...")
        if user_question:
            prompt = get_prompt_template(templateName="docs")
            ollama = get_llm()
            match = vector_store.similarity_search(query=user_question, k=4)
            print(match)
            
            chain = prompt | ollama
            response = chain.invoke({"query": user_question, "documents": match })
            st.write(response)

def invoice_reader_llama():

    files: list = []
    document: str = ''
    with st.sidebar:
        st.info(body="This is Invoice reader application. Please upload a file to get results.", icon=":material/description:")
        files = st.file_uploader(label="Upload files", type="pdf", accept_multiple_files=True)
    
    if len(files) != 0:
        for file in files:
            document += invoice_reader_pypdf(file=file)
        # Splitter and Chunks
        text_splitter = get_splitter()
        chunks = text_splitter.split_text(document)
        # Embeddings
        embeddings=GPT4AllEmbeddings()
        # Vectors
        vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        # Predefined buttons
        with st.sidebar:
            with st.container(border=True):
                st.button(label="Invoice Delays", on_click=process_llama_model, args=['table', vector_store], use_container_width=True)
                st.button(label="Invoice Details", on_click=process_llama_model, args=['old_invoice', vector_store], use_container_width=True)
                st.button(label="Ordered Items", on_click=process_llama_model, args=['all_items', vector_store], use_container_width=True)
                # st.button(label="Bar chart", on_click=process_llama_model, args=['chart', vector_store], use_container_width=True)

def invoice_reader_openai():

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

            #define the LLM
            llm = ChatOpenAI(
                openai_api_key = config["open_ai_api_key"],
                temperature = 0,
                max_tokens = 1000,
                model_name = "babbage-002"
            )

            #output results
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = match, question = user_question)
            st.write(response)

def chat_bot_ollama():

    config: json = read_config()
    ollama = Ollama(base_url=config['local_host'], model='llama3.2')
    with st.sidebar:
        st.info(body="This is normal chatbot based on model llama2. You can ask a question to get the answer", icon=":material/description:")
    user_question = st.text_input(label="Type your questions here")
    if user_question:
        result = ollama.invoke(user_question)
        st.write(result)

def chat_bot_ollama_history():

    with st.sidebar:
        st.info(body="This is normal chatbot based on model llama2. You can ask a question to get the answer", icon=":material/description:")
    config: json = read_config()
    ollama = Ollama(base_url=config['local_host'], model='llama3.2')
    prompt = get_prompt_template(templateName="chat")
    chain = prompt | ollama
    st.write(f'This is the Ollama local chat bot, if you want to quit, type \'exit\'')
    # user_question = st.chat_input("User: ")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

       
    if user_question := st.chat_input(placeholder="User: "):
        if user_question.lower() == "exit":
            st.session_state.messages = []
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        else:
            st.chat_message("user").markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})

            result = chain.invoke({"context": str(st.session_state.messages), "question": user_question})
            with st.chat_message("assistant"):
                st.markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})
            print(str(st.session_state.messages))

def read_config():
    
    with open ('C:/Users/Programming/Documents/Python/AI/Invoice_model/env.json', 'r') as file:
        config = json.load(file)
    return config

def process_llama_model(key: str, vector_store: FAISS):

    ollama: Ollama
    config = read_config()
    with open (config['questions'], 'r') as file:
        questions = json.load(file)
    question = questions[key]
    print(f'User selected button with key: \n{key}, \nquestion: \n{question}')

    if question:
        prompt = get_prompt_template(templateName="docs")
        ollama = get_llm()
        match = vector_store.similarity_search(query=question, k=6)
        chain = prompt | ollama
        response = chain.invoke({"query": question, "documents": match })
        #execute function to manipulate the data
        get_data_transformer_function(func_key=key, ai_response=response)


page_names_to_funcs = {
    "Information": intro,
    "Ollama Documents": invoice_reader_llama,
    "Ollama Documents Custom": invoice_reader_llama_manual,
    "Ollama Chatbot": chat_bot_ollama,
    "Ollama Chatbot History": chat_bot_ollama_history
}

demo_name = st.sidebar.selectbox("Choose an application", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()