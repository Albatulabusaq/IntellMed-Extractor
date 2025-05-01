import os
cache_dir = "D:\\huggingface"
os.environ["TRANSFORMERS_CACHE"] = cache_dir
from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
import spacy
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModel
import re
import torch
import numpy as np
from docx import Document
import PyPDF2
import re
from nltk.tokenize import sent_tokenize
from werkzeug.utils import secure_filename
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# ✅ تحميل الموديلات

# تحميل المحول (tokenizer) والموديل لـ T5
tokenizer = T5Tokenizer.from_pretrained('t5-large')
model = T5ForConditionalGeneration.from_pretrained('t5-large')

# تحميل موديل spaCy
nlp = spacy.load("en_core_web_lg")
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM
)

tokenizers_models = {
    # تصنيف
    "Bio_ClinicalBERT": (
        AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT"),
        AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    ),
    "Longformer": (
        AutoTokenizer.from_pretrained("allenai/longformer-base-4096"),
        AutoModelForSequenceClassification.from_pretrained("allenai/longformer-base-4096")
    ),
    
    # تلخيص
    "BART": (
        AutoTokenizer.from_pretrained("facebook/bart-large-cnn"),
        AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    ),
    "T5": (
        AutoTokenizer.from_pretrained("t5-base"),
        AutoModelForSeq2SeqLM.from_pretrained("t5-base")
    ),

    # توليد
    "BioGPT": (
        AutoTokenizer.from_pretrained("microsoft/biogpt"),
        AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
    ),
    "GPTNeo": (
        AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M"),
        AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    )
}

# تحديد نوع كل موديل
model_task_type = {
    "Bio_ClinicalBERT": "classification",
    "Longformer": "classification",
    "BART": "summarization",
    "T5": "summarization",
    "BioGPT": "generation",
    "GPTNeo": "generation"
}
# تحديد نوع كل موديل (ضروري عشان نعرف وش نستخدم معه generate)


def get_model(model_name):
    return tokenizers_models[model_name]

# ✅ دالة للتلخيص (تستخدم فقط الموديلات اللي تدعم generate)
def summarize_text(text, model_name="BART"):
    if model_task_type[model_name] != "summarization":
        raise ValueError(f"Model '{model_name}' is not suitable for summarization.")
    
    tokenizer, model = get_model(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# ✅ دالة للحصول على التضمينات embeddings (مفيدة للتصنيف أو التحليل)
def get_embeddings(text, model_name="Bio_ClinicalBERT"):
    if model_task_type[model_name] != "classification":
        raise ValueError(f"Model '{model_name}' is not suitable for embeddings.")
    
    tokenizer, _ = get_model(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy()  # [CLS] token
    return cls_embedding

# ✅ الاتصال بقاعدة البيانات
def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS User (
        User_id INTEGER PRIMARY KEY AUTOINCREMENT,
        User_name TEXT NOT NULL,
        Email TEXT NOT NULL UNIQUE,
        Password TEXT NOT NULL
    )''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS Extracted_Text (
        Text_ID INTEGER PRIMARY KEY AUTOINCREMENT,
        User_id INTEGER,
        Text_Content TEXT NOT NULL,
        FOREIGN KEY(User_id) REFERENCES User(User_id)
    )''')

    conn.commit()
    conn.close()

init_db()

# ✅ دوال مساعدة

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    return text

# دالة تلخيص النصوص
# احذف هذه الدالة لأنها مكررة وتسبب تعارضًا.
# دالة التلخيص المعدلة
def summarize_text(text, model_name="BART"):
    tokenizer, model = get_model(model_name)  # احملي النموذج بناءً على model_name
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model.generate(inputs['input_ids'], max_length=150, num_beams=5, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary



def extract_pico(text):
    participants, interventions, comparisons, outcomes = [], [], [], []

    sentences = sent_tokenize(text)

    for sent in sentences:
        sent_lower = sent.lower()

        if any(keyword in sent_lower for keyword in [
            'participants', 'patients', 'subjects', 'individuals', 'cohort',
            'population', 'group of patients', 'cases', 'recruited', 'enrolled',
            'volunteers', 'diagnosed with', 'suffering from', 'affected by',
            'underwent', 'middle-aged adults', 'children', 'adolescents', 'elderly',
            'sample', 'respondents'
        ]):
            participants.append(sent)

        if any(keyword in sent_lower for keyword in [
            'intervention', 'treatment', 'therapy', 'procedure', 'administered',
            'received', 'underwent', 'given', 'exposed to', 'assigned to',
            'medication', 'drug', 'surgery', 'program', 'training', 'protocol',
            'dosage', 'regimen', 'vaccine', 'application of', 'delivery of',
            'supplement', 'diet', 'exercise', 'behavioral intervention',
            'clinical trial'
        ]):
            interventions.append(sent)

        if any(keyword in sent_lower for keyword in [
            'control group', 'placebo', 'no treatment', 'standard care',
            'usual care', 'compared to', 'compared with', 'versus', 'vs.',
            'did not receive', 'alternative treatment', 'non-intervention',
            'reference group', 'baseline group', 'in contrast', 'comparison group'
        ]):
            comparisons.append(sent)

        if any(keyword in sent_lower for keyword in [
            'results', 'outcome', 'effect', 'impact', 'improvement', 'reduction',
            'increase', 'decrease', 'change', 'measured', 'observed',
            'evaluated', 'significant difference', 'benefit', 'response',
            'primary outcome', 'secondary outcome', 'score', 'assessment',
            'endpoint', 'efficacy', 'safety', 'rate', 'incidence', 'mortality',
            'recovery', 'relapse', 'complications', 'success rate', 'progression',
            'clinical outcome', 'duration', 'quality of life'
        ]):
            outcomes.append(sent)

    return {
        'Participants': participants,
        'Intervention': interventions,
        'Comparison': comparisons,
        'Outcome': outcomes
    }




def extract_text_from_docx(file_content):
    doc = Document(BytesIO(file_content))
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text

def extract_text_from_pdf(file_content):
    text = ""
    reader = PyPDF2.PdfReader(BytesIO(file_content))
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                print(f"Page {page_num + 1} is empty or cannot be read properly.")
        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
            continue
    return text.strip()


@app.route('/')
def home_page():
    return render_template('Home1.html')

@app.route('/aboutus')
def aboutus_page():
    return render_template('Aboutus.html')

@app.route('/login')
def login_page():
    return render_template('LoginOne.html')

@app.route('/signin')
def signin_page():
    return render_template('SigninOne.html')

@app.route('/signup', methods=['POST'])
def signup():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO User (User_name, Email, Password) VALUES (?, ?, ?)", (username, email, password))
        conn.commit()
        flash("Account created successfully. Please login.", "success")
    except sqlite3.IntegrityError:
        flash("Username or Email already exists.", "danger")
    conn.close()
    return redirect(url_for('login_page'))

@app.route('/login', methods=['POST'])
def login():
    user_id = request.form['user_id']
    password = request.form['password']

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM User WHERE User_name=? AND Password=?", (user_id, password))
    user = cursor.fetchone()
    conn.close()

    if user:
        session['user_id'] = user_id
        return redirect(url_for('upload_page'))
    else:
        flash("Invalid credentials.", "danger")
        return redirect(url_for('login_page'))

@app.route('/upload')
def upload_page():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))
    username = session.get('user_id')
    return render_template('UploadpageOne.html', username=username) 


# ✅ تحميل الموديلات


@app.route('/process', methods=['POST'])
def process_input():
    if 'user_id' not in session:
        return redirect(url_for('login_page'))

    text_input = request.form.get('text_box')
    file_input = request.files.get('file')

    if file_input and file_input.filename != '':
        file_content = file_input.read()
        if file_input.filename.endswith('.docx'):
            pdf_text = extract_text_from_docx(file_content)
        elif file_input.filename.endswith('.pdf'):
            pdf_text = extract_text_from_pdf(file_content)
        else:
            flash("Unsupported file format! Only .pdf or .docx allowed.", "danger")
            return redirect(url_for('upload_page'))
    elif text_input and text_input.strip():
        pdf_text = text_input.strip()
    else:
        flash("Please enter some text or upload a file!", "danger")
        return redirect(url_for('upload_page'))

    # تسجيل النص المستخرج
    print(f"Extracted text: {pdf_text}")  # تأكد من أن النص يتم استخراجه بشكل صحيح
    
    pdf_text = clean_text(pdf_text)

    if not pdf_text:
        flash("No text extracted from the input!", "danger")
        return redirect(url_for('upload_page'))

    summary = summarize_text(pdf_text)
    pico = extract_pico(pdf_text)

    session['original_text'] = pdf_text  # حفظ النص الأصلي
    session['summary'] = summary
    session['pico'] = pico

    return redirect(url_for('results_page'))


@app.route('/results')
def results_page():
    return render_template(
        'Resultspage1p.html',
        input_text=session.get('input_text', ''),
        summary=session.get('summary', ''),
        pico=session.get('pico', {})
    )



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login_page'))

if __name__ == '__main__':
    app.run(debug=True)
