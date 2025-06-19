from flask import Flask, render_template, request, redirect, url_for
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Setup
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

app = Flask(__name__)

# Load model and vectorizer
with open('spam_models_bundle.pkl', 'rb') as f:
    model_bundle = pickle.load(f)

vectorizer = model_bundle['vectorizer']
nb_model = model_bundle['naive_bayes']

# Clean text function (once, globally)
def clean_text(text):
    text = re.sub(r'\W', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

# Reusable spam prediction function
def predict_spam(message):
    cleaned_message = clean_text(message)
    vectorized = vectorizer.transform([cleaned_message])
    prediction = nb_model.predict(vectorized)[0]
    probability = nb_model.predict_proba(vectorized)[0][1]
    label = 'Spam' if prediction == 1 else 'Not Spam'
    return label, probability

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/detector', methods=['GET', 'POST'])
def detector():
    result = None
    probability = None
    message = ""
    if request.method == 'POST':
        message = request.form['message']
        result, probability = predict_spam(message)
    return render_template('detector.html', result=result, probability=probability, message=message)

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    result, probability = predict_spam(message)
    return render_template('predict.html', prediction=result, probability=probability, message=message)


@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        services = request.form['services']
        # Save or handle form data here
        return redirect(url_for('home'))
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)
