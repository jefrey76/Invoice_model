from invoice2data import extract_data
import PyPDF2
import pypdf
import os

def invoice_reader_pypdf(file):

    text = ''
    fileReader = pypdf.PdfReader(file)
    print(fileReader.pages[0].extract_text())
    for page in fileReader.pages:
        text += page.extract_text()
    return text

def test():
    pdfFiles = "C:\\Users\\Programming\\documents\\Python\AI\\Invoice_model\\PDF_files\\order_confirmation_1.pdf"
    filePath = os.path.join(pdfFiles, 'order_confirmation_1.pdf')
    result = pypdf.PdfReader(pdfFiles)
    print(result.pages[0].extract_text())
