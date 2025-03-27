import os
from flask import Flask, request, render_template
from summarizer import extractive_summary, abstractive_summary_t5, extract_text_from_pdf, extract_text_from_docx

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    text = ""

    if request.method == "POST":
        method = request.form.get("method")  # Extractive or Abstractive
        
        # Handling text input
        if request.form["text"]:
            text = request.form["text"]

        # Handling file upload
        if 'file' in request.files:
            uploaded_file = request.files['file']
            if uploaded_file.filename != "":
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)

                # Extract text from PDF or DOCX
                if uploaded_file.filename.endswith(".pdf"):
                    text = extract_text_from_pdf(file_path)
                elif uploaded_file.filename.endswith(".docx"):
                    text = extract_text_from_docx(file_path)
                else:
                    summary = "Unsupported file format. Please upload a PDF or DOCX."

        # Generate Summary
        if text:
            if method == "extractive":
                summary = extractive_summary(text)
            elif method == "abstractive":
                summary = abstractive_summary_t5(text)
        elif summary == "":
            summary = "No text found for summarization."

    return render_template("index.html", text=text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
