## Functions used in Projet_Airport


# Libraries 

# Standard libraries
import re
import string
import time

# Data manipulation and visualization
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from wordcloud import WordCloud

# Text processing
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Language detection and translation
from langdetect import detect
from deep_translator import GoogleTranslator

# Feature extraction and topic modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Spacy for NLP
import spacy

# Gensim for topic modeling and word embeddings
import gensim
import gensim.corpora as corpora
from gensim.models import TfidfModel, LdaModel, CoherenceModel

# Summarization with Sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# BERT and transformers
import torch



# Exploratoty data analysis ------------------------------------------------------

def value_count_df(df):

    nbr_col_aff=0
    for i in range (1, len(df.columns)):
        if df.iloc[:,i].nunique()<1000:
            print("column", i,"-----", df.iloc[:,i].unique(), "\n")
            print(df.iloc[:,i].value_counts(), "\n")
            nbr_col_aff +=1
    return (nbr_col_aff, "columns were printed ")



def hist_variables(df):

    for i in range (1, len(df.columns)):
        if df.iloc[:,i].nunique()<500:
                    fig = px.histogram(df, x=df.iloc[:,i], title=df.columns[i])
                    fig.show()

# Data pre-processing -------------------------------------------------------------

def few_cara(text):

    if len(text.split()) <= 1 or len(text) < 12:   # to delete short comments that are useless
        return 'NaN'  
    return text  



def trad(text):

    # Create a dictionary to store translations in cache
    translation_cache = {}

    # Check if the translation is already in cache
    if text in translation_cache:
        return translation_cache[text]
    else:
        # Check if the text is already in English
        if detect(text) == 'en':
            # No need to translate here
            translation_cache[text] = text
            return text
        else:
            # Add a pause to avoid overloading the API
            time.sleep(0.1)
            # Translate the text
            translated_text = GoogleTranslator(source='auto', target='en').translate(text)
            # Store the translation in cache
            translation_cache[text] = translated_text
            return translated_text



def clean_text(text):    

    # Deleting special characters and punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    #regex numbers 
    text = re.sub(r"\b\d+(\.\d+)?\b", "NUMBER", text) #figures not followed directly by text characters
    text = re.sub(r"\b\d+(\.\d+)?\b\W?", "NUMBER ", text)  #figures followed directly by text characters 

    # Converting to lower case
    text = text.lower()
    
    # Text tokenisation
    tokens = word_tokenize(text)
    
    # Stop words
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    
    #lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in tokens]

    # Join lemmatised tokens into a single string
    cleaned_text = ' '.join(lemmatized_words)

    return cleaned_text


# Wordcloud Analysis ------------------------------------------------------------------------------------

def generate_wordcloud(df, titre):

    all_text = ' '.join(df)
    list = ["airport", "flight", "Edinburgh"]  #words that are over represented but do not interest us, that are not relevant 
    for elmt in list: 
        all_text = all_text.replace(elmt ,'')
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
    plt.figure(figsize=(6, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(titre, fontsize=14)
    plt.axis('off')
    plt.show()


# Topic Modelling ----------------------------------------------------------------------------------------

# TF-IDF x LDA using Sklearn library -------------------------------------------------------------------

def generate_TFIDF(df_column, ngram_range):
    
    vectorizer  = TfidfVectorizer(ngram_range=ngram_range)
    features = vectorizer.fit_transform(df_column)

    features = features.toarray() # Convert characteristics into a hollow matrix representation
    
    return features, vectorizer


def generate_LDA(components, feature, vectorizer, num_top_words, randomstate):

    lda = LatentDirichletAllocation(n_components=components, random_state=randomstate)
    lda.fit(feature)
    
    # Obtain word distributions for each subject
    topic_word_distributions = lda.components_
    
    # Get the vocab
    vocab = vectorizer.get_feature_names_out()
    
    # List of words to exclude
    list2 = ["airport", "flight", "edinburgh", "number"]
    
    # Accumulate results
    results = []
    
    # For each subject
    for topic_idx, topic in enumerate(topic_word_distributions):
        # Get indexes of the most likely words for the subject
        top_word_indices = topic.argsort()[-num_top_words:][::-1]
        # Get the associated words
        top_words = [vocab[i] for i in top_word_indices]
        # Exclude unwanted words
        filtered_top_words = [word for word in top_words if word not in list2]
        # Accumulate the result
        results.append(f"Topic {topic_idx}: {' '.join(filtered_top_words)}")
    
    # Return all topics
    return '\n'.join(results)


# TF-IDF x LDA using Gensim library ---------------------------------------------------------------------------

def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    # Load the small English language model from spaCy, disabling the parser and NER components
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    
    # Process multiple documents in one go with nlp.pipe to improve performance
    docs = list(nlp.pipe(texts, disable=["parser", "ner"]))  
    
    # Initialize an empty list to hold the processed texts
    texts_out = []
    
    # Iterate over each processed document
    for doc in docs:
        # Initialize a list to hold the lemmatized tokens for the current document
        new_text = []
        
        # Iterate over each token in the document
        for token in doc:
            # Check if the token's part of speech (POS) is in the allowed postags
            if token.pos_ in allowed_postags:
                # Append the lemmatized form of the token to the new_text list
                new_text.append(token.lemma_)
        
        # Join the lemmatized tokens into a single string
        final = " ".join(new_text)
        
        # Append the processed text to the texts_out list
        texts_out.append(final)
    
    # Return the list of processed texts
    return texts_out


def gen_words(text):
    # Initialize an empty list to hold the processed tokens
    final = []
    
    # Iterate over each token in the input text
    for token in text:
        # Use gensim's simple_preprocess function to tokenize the text
        # deacc=True removes accent marks from the tokens
        new = gensim.utils.simple_preprocess(token, deacc=True)
        
        # Append the processed token to the final list
        final.append(new)
    
    # Return the list of processed tokens
    return final



def make_bigrams(texts, bigram):
    return([bigram[doc] for doc in texts])


def make_trigrams(texts, bigram, trigram):
    return ([trigram[bigram[doc]] for doc in texts])


def process_corpus(data_bigrams_trigrams, low_value=0.03):
    # Create the dictionary
    id2word = corpora.Dictionary(data_bigrams_trigrams)
    
    # Preprocessed text
    texts = data_bigrams_trigrams
    
    # Create the bag-of-words corpus
    corpus = [id2word.doc2bow(text) for text in texts]
    
    # Create the TF-IDF model
    tfidf = TfidfModel(corpus, id2word=id2word)
    
    # Variables to store low TF-IDF value words and those missing in TF-IDF
    words = []
    words_missing_in_tfidf = []
    
    # Filter out low TF-IDF value words from the corpus
    for i in range(len(corpus)):
        bow = corpus[i]
        low_value_words = []  # Reinitialize for each document
        tfidf_ids = [id for id, value in tfidf[bow]]
        bow_ids = [id for id, value in bow]
        
        # Identify low TF-IDF value words
        low_value_words = [id for id, value in tfidf[bow] if value < low_value]
        drops = low_value_words + words_missing_in_tfidf
        
        # Add low TF-IDF value words to the list of words to remove
        for item in drops:
            words.append(id2word[item])
        
        # Words missing in TF-IDF (those with a TF-IDF score of 0)
        words_missing_in_tfidf = [id for id in bow_ids if id not in tfidf_ids]
        
        # Create a new bag-of-words without low TF-IDF value words
        new_bow = [b for b in bow if b[0] not in low_value_words and b[0] not in words_missing_in_tfidf]
        corpus[i] = new_bow

    return corpus, words, texts, id2word


def train_lda_model(corpus, id2word, num_topics):
    # Train an LDA model with the given parameter 
    lda_model = LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics)
    return lda_model


def compute_coherence(lda_model, corpus, texts, coherence='c_v'):
    # Calculate the consistency of subjects to evaluate the model
    coherence_model = CoherenceModel(model=lda_model, texts=texts, corpus=corpus, coherence=coherence)
    return coherence_model.get_coherence()


def grid_search_lda(corpus, texts, id2word, param_grid, coherence='c_v'):
    best_coherence = -1
    best_model = None

    for num_topics in param_grid['num_topics']:
        lda_model = train_lda_model(corpus, id2word, num_topics)

        # Calculate the consistency of subjects to evaluate the model
        coherence_score = compute_coherence(lda_model, corpus, texts, coherence)

        # Update best model 
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_model = lda_model

    return best_model, best_coherence



# BERT ---------------------------------------------------------------------------------------------------------

def Sentiment_anlysis(sentence, tokenizer, model):
    # Function which calculates a sentiment score between 1 to 5 (very negative to very positive) 

    encoder = tokenizer.encode(sentence, return_tensors='pt')
    result = model(encoder)
    
    return int(torch.argmax(result.logits))+1

# Word2vec -----------------------------------------------------------------------------------------------------

def preprocess_corpus(df):

    lem = WordNetLemmatizer()

    # Convert to list
    corpus = df.tolist()
    
    # Convert to lowercase
    corpus = [doc.lower() for doc in corpus]
    
    # Remove punctuation
    punctuations = list(string.punctuation)
    corpus = ["".join([char for char in list(doc) if not (char in punctuations)]) for doc in corpus]
    
    # Tokenization
    corpus_tk = [word_tokenize(doc) for doc in corpus]
    
    # Lemmatization
    corpus_lm = [[lem.lemmatize(word) for word in doc] for doc in corpus_tk]
    
    # Remove stopwords
    empty_words = stopwords.words('english')
    corpus_sw = [[word for word in doc if not (word in empty_words)] for doc in corpus_lm]
    
    # Remove words with less than 3 characters
    corpus_sw = [[word for word in doc if len(word) >= 3] for doc in corpus_sw]
    
    # Remove empty lists
    # corpus_sw_no_empty = [lst for lst in corpus_sw if lst]
    
    return corpus_sw

def my_doc_2_vec(doc,trained):

# function for transforming a document into a vector from its tokens
# input: the document to be processed and the pre-trained model
# output: vector representing the document

    # representation dimension
    p = trained.vectors.shape[1]
    # initialise the vector
    vec = np.zeros(p)
    # number of tokens found
    nb = 0
    # process each token in the document
    for tk in doc:
        # process only recognised tokens
        try:
            values = trained[tk]
            vec = vec + values
            nb = nb + 1.0
        except:
            pass
    # average the values
    # only if recognised tokens have been found, of course
    if (nb > 0.0):
        vec = vec/nb
    # return the vector
    # if no tokens found, we have a vector of zero values
    return vec


def generate_summary(text, sentence_count=5):
    # Initialize the PlaintextParser with the input text and English tokenizer
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    
    # Initialize the Stemmer for English
    stemmer = Stemmer("english")
    
    # Initialize the LSA Summarizer with the stemmer
    summarizer = Summarizer(stemmer)
    
    # Set the stop words for the summarizer
    summarizer.stop_words = get_stop_words("english")
    
    # Initialize an empty list to hold the summary sentences
    summary = []
    
    # Generate the summary by iterating over the sentences returned by the summarizer
    for sentence in summarizer(parser.document, sentence_count):
        # Convert each sentence to string and append to the summary list
        summary.append(str(sentence))
    
    # Join the summary sentences into a single string and return
    return ' '.join(summary)
