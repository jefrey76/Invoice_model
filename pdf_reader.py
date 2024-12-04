from invoice2data import extract_data
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
import pypdf


def invoice_reader_pypdf(file):

    text = ''
    fileReader = pypdf.PdfReader(file)
    for page in fileReader.pages:
        text += page.extract_text()
    return text

def document_loader(files: list):
    
    documents: list = []
    for file in files:
        print(dir(file))
        loader = PyPDFLoader(file.name)
        loaded_file = loader.load()
        documents.append(loaded_file)
    return documents
