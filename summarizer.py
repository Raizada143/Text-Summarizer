import fitz  # PyMuPDF for PDF
import docx  # For Word files
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

# Load T5 Model and Tokenizer (Global Initialization for Faster Processing)
t5_model_name = "t5-small"  # You can use "t5-base" or "t5-large" for better results
t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name)

# Extractive Summarization using Sumy (LSA Algorithm)
def extractive_summary(text, num_sentences=3):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join(str(sentence) for sentence in summary)

# Abstractive Summarization using T5
def abstractive_summary_t5(text, max_length=150):
    input_text = "summarize: " + text
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=False)
    
    summary_ids = t5_model.generate(
        inputs, max_length=max_length, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    
    summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Extract text from PDF files
import fitz

def extract_text_from_pdf(file_path):
    try:
        doc = fitz.open(file_path)  # Open PDF
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        if not text.strip():
            return "No extractable text found in the PDF."
        return text
    except fitz.EmptyFileError:
        return "Error: The uploaded PDF is empty."



# Extract text from Word files
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])
