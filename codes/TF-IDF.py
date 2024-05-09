import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def tfidf_summarize(text, num_sentences):
    # Tokenize the text 
    sentences = nltk.sent_tokenize(text)
    vect = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vect.fit_transform(sentences)
    
    # Calculate the average TF-IDF score for each sentence
    tfidf_sentence = np.mean(tfidf_matrix.toarray(), axis=1)
    
    # Get the top-ranked sentences
    top_idx = tfidf_sentence.argsort()[-num_sentences:][::-1]
    top_idx.sort()
    summary = ' '.join([sentences[i] for i in top_idx])
    
    return summary

# Perform extractive text summarisation
with open('football.txt', 'r') as file:
    text = file.read()

print(tfidf_summarize(text,3))

