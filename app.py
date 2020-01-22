import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import nltk
import re
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

CONTRACTION_MAP = {"ain't": 'is not', "aren't": 'are not', "can't": 'cannot', "can't've": 'cannot have', "'cause": 'because', "could've": 'could have', "couldn't": 'could not', "couldn't've": 'could not have', "didn't": 'did not', "doesn't": 'does not', "don't": 'do not', "hadn't": 'had not', "hadn't've": 'had not have', "hasn't": 'has not', "haven't": 'have not', "he'd": 'he would', "he'd've": 'he would have', "he'll": 'he will', "he'll've": 'he he will have', "he's": 'he is', "how'd": 'how did', "how'd'y": 'how do you', "how'll": 'how will', "how's": 'how is', "I'd": 'I would', "I'd've": 'I would have', "I'll": 'I will', "I'll've": 'I will have', "I'm": 'I am', "I've": 'I have', "i'd": 'i would', "i'd've": 'i would have', "i'll": 'i will', "i'll've": 'i will have', "i'm": 'i am', "i've": 'i have', "isn't": 'is not', "it'd": 'it would', "it'd've": 'it would have', "it'll": 'it will', "it'll've": 'it will have', "it's": 'it is', "let's": 'let us', "ma'am": 'madam', "mayn't": 'may not', "might've": 'might have', "mightn't": 'might not', "mightn't've": 'might not have', "must've": 'must have', "mustn't": 'must not', "mustn't've": 'must not have', "needn't": 'need not', "needn't've": 'need not have', "o'clock": 'of the clock', "oughtn't": 'ought not', "oughtn't've": 'ought not have', "shan't": 'shall not', "sha'n't": 'shall not', "shan't've": 'shall not have', "she'd": 'she would', "she'd've": 'she would have', "she'll": 'she will', "she'll've": 'she will have', "she's": 'she is', "should've": 'should have', "shouldn't": 'should not', "shouldn't've": 'should not have', "so've": 'so have', "so's": 'so as', "that'd": 'that would', "that'd've": 'that would have', "that's": 'that is', "there'd": 'there would', "there'd've": 'there would have', "there's": 'there is', "they'd": 'they would', "they'd've": 'they would have', "they'll": 'they will', "they'll've": 'they will have', "they're": 'they are', "they've": 'they have', "to've": 'to have', "wasn't": 'was not', "we'd": 'we would', "we'd've": 'we would have', "we'll": 'we will', "we'll've": 'we will have', "we're": 'we are', "we've": 'we have', "weren't": 'were not', "what'll": 'what will', "what'll've": 'what will have', "what're": 'what are', "what's": 'what is', "what've": 'what have', "when's": 'when is', "when've": 'when have', "where'd": 'where did', "where's": 'where is', "where've": 'where have', "who'll": 'who will', "who'll've": 'who will have', "who's": 'who is', "who've": 'who have', "why's": 'why is', "why've": 'why have', "will've": 'will have', "won't": 'will not', "won't've": 'will not have', "would've": 'would have', "wouldn't": 'would not', "wouldn't've": 'would not have', "y'all": 'you all', "y'all'd": 'you all would', "y'all'd've": 'you all would have', "y'all're": 'you all are', "y'all've": 'you all have', "you'd": 'you would', "you'd've": 'you would have', "you'll": 'you will', "you'll've": 'you will have', "you're": 'you are', "you've": 'you have'}


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    # re.compile(regex).search(subject) is equivalent to re.search(regex, subject).
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = re.sub("’", "'", text)
    expanded_text = contractions_pattern.sub(expand_match, expanded_text)

    return expanded_text

# Function to Preprocess the Reviews
def clean_doc(doc):
    # Removing contractions
    doc = expand_contractions(doc)

    # split into tokens by white space
    tokens = doc.split(' ')

    # Converting into lower case
    tokens = [w.lower() for w in tokens]

    # remove special characters from each token
    tokens = [re.sub(r"[^-a-zA-Z#\s]", '', i) for i in tokens]
    tokens = [re.sub(r"[\r\n]", '', i) for i in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]

    # lemmatizing
    lmtzr = nltk.stem.WordNetLemmatizer()
    tokens = [lmtzr.lemmatize(w) for w in tokens]

    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 1]
    return tokens

app = Flask(__name__)
model = joblib.load('svm_model.pkl')
vocab_dict = pickle.load(open('vocab_dict.pkl', 'rb'))

labels = ['car_repair_appointments', 'making_restaurant_reservations', 'movie_tickets', 'ordering_coffee',
          'ordering_pizza', 'taxi_service']

tfidf_vectorizer = TfidfVectorizer(max_df=0.90,max_features=1000,stop_words='english',vocabulary=vocab_dict)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    command_input1 = [x for x in request.form.values()]
    print(command_input1)
    command_input2 = ' '.join(clean_doc(command_input1[0]))
    print(command_input2)
    command_input3 = pd.core.series.Series(command_input2)
    final_features = tfidf_vectorizer.fit_transform(command_input3)
    prediction = model.predict(final_features)
    output = labels[prediction[0]]
    print(output)
    return render_template('index.html', prediction_text=output)

@app.route('/predict_api',methods=['POST'])
def predict_api():
  
    # For direct API calls through request

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(debug=True,)
