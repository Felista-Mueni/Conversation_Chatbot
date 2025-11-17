import nltk
import streamlit as st
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# --- Safe NLTK downloads ---
nltk_packages = ['punkt', 'averaged_perceptron_tagger', 'stopwords', 'wordnet']

for pkg in nltk_packages:
    try:
        # Check if package is already downloaded
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg)

# --- Load your text file (must be in same folder) ---
with open('chatbot.txt', 'r', encoding='utf-8') as f:
    data = f.read().replace('\n', ' ')

sentences = sent_tokenize(data)

# --- Preprocessing function ---
def preprocess(sentence):
    words = word_tokenize(sentence)
    words = [
        word.lower()
        for word in words
        if word.lower() not in stopwords.words('english')
        and word not in string.punctuation
    ]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return words

corpus = [preprocess(sentence) for sentence in sentences]

# --- Find most relevant sentence ---
def get_most_relevant_sentence(query):
    query = preprocess(query)
    max_similarity = 0
    most_relevant_sentence = ""

    for sentence in corpus:
        similarity = len(set(query).intersection(sentence)) / float(len(set(query).union(sentence)))
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_sentence = " ".join(sentence)

    return most_relevant_sentence

def chatbot(question):
    return get_most_relevant_sentence(question)

# --- Streamlit app ---
def main():
    st.title("Chatbot")
    st.write("Hello! I'm a chatbot. Ask me anything about the topic in the text file.")

    question = st.text_input("You:")

    if st.button("Submit"):
        response = chatbot(question)
        st.write("Chatbot: " + response)

if __name__ == "__main__":
    main()
