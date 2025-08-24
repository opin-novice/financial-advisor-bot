import PyPDF2
import sys

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

if __name__ == "__main__":
    pdf_path = "LegalRAG.pdf"
    try:
        text = extract_text_from_pdf(pdf_path)
        with open("LegalRAG_content.txt", "w", encoding="utf-8") as f:
            f.write(text)
        print("Text extracted successfully to LegalRAG_content.txt")
    except Exception as e:
        print(f"Error extracting text: {e}")
        sys.exit(1)