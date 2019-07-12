import nltk
import numpy as np
import random
import string # to process standard python strings
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Reading in the corpus
with open('corpus/chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

#Tokenisation
sent_tokens = nltk.sent_tokenize(raw) #converts to list of sentences 
word_tokens = nltk.word_tokenize(raw) #converts to list of words

# Preprocessing
lemma = WordNetLemmatizer()

def lem_tokens(tokens):
    return [lemma.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_normalizer(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Keyword Matching
GREETING_INPUTS = ('hi', 'hello', 'greeting', 'sup', 'hey', 'what\'s up')
GREETING_RESPONSES = ['hi', 'hey', '*nods*', 'hi there', 'hello', 'I am glad! Ypu are talking to me']

def bot_greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


# Generating response
def response(user_response):
    bot_response = ''
    sent_tokens.append(user_response)
    frequency_vector = TfidfVectorizer(tokenizer=lem_normalizer, stop_words='english')
    tf_idf = frequency_vector.fit_transform(sent_tokens)
    vals = cosine_similarity(tf_idf[-1], tf_idf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        bot_response = bot_response+'sorry, I can\'t seem to understand you'
        return bot_response
    else:
        bot_response = bot_response+sent_tokens[idx]
        return bot_response

flag=True
print('FejiBot: Hello! my name is Feji. I will answer your queries about chatbots.'
      'If you want to exit, type Bye!')

while(flag == True):
    user_response = input()
    user_response = user_response.lower()
    if(user_response != 'bye'):
        if(user_response is 'thanks' or user_response is 'thank you'):
            flag=False
            print('FejiBot: You are welcome..')
        else:
            if(bot_greeting(user_response) != None):
                print('FejiBot: '+bot_greeting(user_response))
            else:
                print('FejiBot: ', end = '')
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print('FejiBot: Bye! Nice chatting with you')
