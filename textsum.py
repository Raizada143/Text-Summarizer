import nltk
import re
import heapq
import flask
from flask import Flask, request, jsonify
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Download necessary NLTK data
nltk.download('punkt')

app = Flask(__name__)

def extractive_summary(text, num_sentences=3):
    """Summarizes text using LexRank algorithm."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, num_sentences)
    return " ".join([str(sentence) for sentence in summary])

@app.route('/summarize', methods=['POST'])
def summarize_text():
    """API endpoint to summarize text."""
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    summary = extractive_summary(text)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)