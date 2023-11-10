import fitz  # PyMuPDF for PDF parsing
from transformers import pipeline  # Hugging Face Transformers library for pre-trained models

# Step 1: Extract text from the PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text

# Step 2: Load pre-trained question-answering model
qa_pipeline = pipeline("question-answering")

# Example PDF file path
pdf_path = "data/Transformerbook.pdf"

# Step 3: Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# User's question
user_question = "What is a Transformer?"

# Step 4: Use the pre-trained model for question answering
result = qa_pipeline(question=user_question, context=pdf_text)

# Step 5: Display the answer
print("Answer:", result["answer"])
print("Score:", result["score"])
