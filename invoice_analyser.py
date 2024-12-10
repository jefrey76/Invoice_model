import streamlit as st
import json
import os
import faiss
from pdf_reader import *
from answer_converter import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore



class Templates:

    template_for_docs = """Answer the question based on the provided invoices. Try to avoid adding introduction and explaining the answer, focus only on exact answer.
        If you will be asked about the table format, provide only table.

        Invoices: {documents}

        Question: {query}

        Answer: """


    template_for_docs_invoice = """Answer the question based on the provided invoices. Try to avoid adding introduction and explaining the answer, focus only on exact answer. 
        Attached documents are invoices type and each of them should contains common fields like: Customer, 
        Items, Quantity, Tax Rate, Total Amount, Order Date. Focus on these fields while analyzing document. To determine how many documents have been 
        implemented, try to recognize invoice numbers. Use the table format while answering.

        Invoices: {documents}

        Question: {query}

        Answer: """


    template_for_chat = """Answer the question below.

        Here is the conversation history: {context}.
        If you can't find the answer based on the history, use your data to provide the details.

        Question: {question}

        Answer
        """
    

class Labels:

    invoice_reader_llama_info = "This is Invoice reader application. Please upload a file to get results."
    chat_bot_ollama_info = "This is chatbot based on model llama3. You can ask a question to get the answer"
    chat_bot_ollama_history_info = "This is chatbot based on model llama3. You can ask a question to get the answer."
    chat_bot_ollama_history_info_exit = f'This is the Ollama local chat bot, if you want to clear conversation history, type \'exit\'.'

def intro():

    st.write("# This is Gen AI Upskill program demo application!")
    st.sidebar.success("Select a function above.")

    st.markdown(
        """

        This application has been created to help you analize invoice documents
        You can select 4 specific functions that uses Gen AI.

        **👈 Select a function from the dropdown on the left** and follow the instructions.

        ### Bellow you can see description of specific functions

        - Ollama Documents - upload documents and use pre-defined buttons to get results
        - Ollama Documents Custom - upload documents and ask your question to get answer
        - Ollama Chatbot - this uses AI model to answer your questions, it does not keep the history
        - Ollama Chatbot History - The chatbot that keep history of your conversation

    """
    )

def get_prompt_template(templateName: str)->PromptTemplate:

    if templateName == "docs":
        template = Templates.template_for_docs
    elif templateName == "chat":
        template = Templates.template_for_chat
    elif templateName == "docs_invoice": 
        template = Templates.template_for_docs_invoice
    
    prompt_template = PromptTemplate.from_template(template)
    return prompt_template

def get_llm()->OllamaLLM:

    config: json = read_config()
    ollama = OllamaLLM(
        base_url=config['local_host'], 
        model=config['model'],
        temperature=config['temperature']
        )
    return ollama

def get_vector_store()->FAISS:

    embeddings = GPT4AllEmbeddings()
    index = faiss.IndexFlatL2(len(embeddings.embed_query("Initial text")))

    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )

    return vector_store

def add_doc_to_vector_store(vector_store: FAISS, document: Document):

    uuids = [str(uuid4()) for _ in range(len(document))]
    vector_store.add_documents(documents=document, ids=uuids)

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
        "get_date_details": get_date_details,
        "get_invoice_details": get_invoice_details,
        "all_items": get_all_items,
        "chart": prepare_bar_chart
        }
    function_dict[func_key](ai_response)

def invoice_reader_llama_manual():

    files: list = []
    # document: str = ''
    vector_store = get_vector_store()
    with st.sidebar:
        st.info(body="This is Invoice reader application. Please upload a file to get results.", icon=":material/description:")
        files = st.file_uploader(label="Upload files", type="pdf", accept_multiple_files=True)
    
    if len(files) != 0:
        for file in files:
            # document += invoice_reader_pypdf(file=file)
            document = document_loader(file)
            uuids = [str(uuid4()) for _ in range(len(document))]
            vector_store.add_documents(documents=document, ids=uuids)

        #-------------------------------------------------------------------------
        # text_splitter = get_splitter()
        # chunks = text_splitter.split_text(document)
        # embeddings=GPT4AllEmbeddings()
        # vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)
        #-------------------------------------------------------------------------

        user_question = st.text_input("Type your question here...")
        if user_question:
            prompt = get_prompt_template(templateName="docs_invoice")
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
        st.info(body=Labels.invoice_reader_llama_info, icon=":material/description:")
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
                st.button(label="Invoice Delays", on_click=process_llama_model, args=['get_date_details', vector_store], use_container_width=True)
                st.button(label="Invoice Details", on_click=process_llama_model, args=['get_invoice_details', vector_store], use_container_width=True)
                st.button(label="Ordered Items", on_click=process_llama_model, args=['all_items', vector_store], use_container_width=True)

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
                model_name = "open_ai"
            )

            #output results
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents = match, question = user_question)
            st.write(response)

def chat_bot_ollama():

    config: json = read_config()
    ollama = OllamaLLM(base_url=config['local_host'], model='llama3.2')
    with st.sidebar:
        st.info(body=Labels.chat_bot_ollama_info, icon=":material/description:")
    user_question = st.text_input(label="Type your questions here")
    if user_question:
        result = ollama.invoke(user_question)
        st.write(result)

def chat_bot_ollama_history():

    with st.sidebar:
        st.info(body=Labels.chat_bot_ollama_history_info, icon=":material/description:")
    config: json = read_config()
    ollama = OllamaLLM(base_url=config['local_host'], model='llama3.2')
    prompt = get_prompt_template(templateName="chat")
    chain = prompt | ollama
    st.info(Labels.chat_bot_ollama_history_info_exit, icon="ℹ️")
 

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
    
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_file_path = os.path.join(root_dir, "env.json")

    with open (config_file_path, 'r') as file:
        config = json.load(file)
    return config

def process_llama_model(key: str, vector_store: FAISS):

    ollama: OllamaLLM
    config = read_config()
    questions_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), config['questions'])
    with open (questions_config, 'r') as file:
        questions = json.load(file)
    question = questions[key]
    print(f'User selected button \nKey: {key}, \nQuestion: {question}')

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

application = st.sidebar.selectbox("Choose an application", page_names_to_funcs.keys())
page_names_to_funcs[application]()
