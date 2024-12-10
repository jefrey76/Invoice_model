from langchain_community.document_loaders import PyPDFLoader
import pypdf
import os

def invoice_reader_pypdf(file):

    text = ''
    fileReader = pypdf.PdfReader(file)
    for page in fileReader.pages:
        text += page.extract_text()
    return text

def documents_loader(files):
    
    documents_path = os.path.expanduser("~/Documents")
    documents = []
    for file in files:

        temp_file = os.path.join(documents_path, file.name)
        with open(temp_file, "wb") as temp:
            temp.write(file.read())
        loader = PyPDFLoader(temp_file)
        loaded_file = loader.load()
        print(loaded_file)
        documents.append(loaded_file)

    return documents

def document_loader(file):

    documents_path = os.path.expanduser("~/Documents")
    temp_file = os.path.join(documents_path, file.name)
    with open(temp_file, "wb") as temp:
        temp.write(file.read())
    loader = PyPDFLoader(temp_file)
    loaded_file = loader.load()

    return loaded_file