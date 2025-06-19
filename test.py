import pickle

# Load the saved model bundle
with open('spam_models_bundle.pkl', 'rb') as f:
    models = pickle.load(f)

# Access individual components
nb_model = models['naive_bayes']
lr_model = models['logistic_regression']
rf_model = models['random_forest']
vectorizer = models['vectorizer']

# Example prediction using Naive Bayes
def clean_text(text):
    import re
    from nltk.corpus import stopwords
    import nltk
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', text)
    text = text.lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)


def predict_spam(message, model):
    cleaned_message = clean_text(message)
    vectorized = vectorizer.transform([cleaned_message])
    prediction = model.predict(vectorized)
    probability = model.predict_proba(vectorized)
    return ("Spam" if prediction[0] == 1 else "Not Spam", probability[0][1])

# Example usage
print(predict_spam("Congratulations! You've won a free ticket to Bahamas! Click here to claim your prize.", nb_model))
print(predict_spam("Hey, are you coming to the party?", lr_model))
