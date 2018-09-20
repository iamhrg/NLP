#This code is entitled to Harsh Gandhi.
#The following code shows some basic steps for starting with NLP. This includes Tokenization, Stemming, Lemmatization, Normalization and Vocabulary extraction mainly.

#Importing the NLP Libraries. Here Inflect module of python is used for performing some of the Normalization functions.
import inflect
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

#The following is a set of documents under examination.
document_collection = [ 'A group of kids is playing in a yard and an old man is standing in the background countries liek USA and UK',
                       'A group of children is playing in the house and there is no man standing in the background',
                       'The young boys are playing outdoors and the man is smiling nearby',
                       'The kids are playing outdoors near a man with a smile',
                       'There is no boy playing outdoors and there is no man smiling',
                       'A group of boys in a yard is playing and a man is standing in the background',
                       'A brown dog is attacking another animal in front of the tall man in pants',
                       'A brown dog is attacking another dog in front of the man in pants',
                       'Two dogs are fighting',
                       'Two dogs are wrestling and hugging']

#The following is n example set of vocabulary dictionary to replace some of the commonly used abbreviations of the English language during the Normalization phase.
abb={
'USA':'United States of America',
'UK':'United Kingdom',
'BA' : 'Bachelor of Arts',
'BS' : 'Bachelor of Science',
'MA' : 'Master of Arts',
'MPHIL' : 'Master of Philosophy',
'JD' : 'Juris Doctor',
'PA' : 'Personal Assistant',
'MD' : 'Managing Director',
'VP' : 'Vice President',
'SVP' : 'Senior Vice President',
'CEO' : 'Chief Executive Officer',
}

tokens = []

#Tokenization of Document Collection
for document in document_collection:
    #Stopwords Removal
    term_frequency_vectorizer = CountVectorizer(stop_words='english')
    term_frequency_model = term_frequency_vectorizer.fit_transform(document_collection)
    
    tokens = tokens + word_tokenize(document)
fd = nltk.FreqDist(tokens)

print ("-"*120)
print ("Total number of Words in the Document Collection: ", len(tokens))
print ("-"*120)
print ("Unique Words Vocabulary:\n\n", term_frequency_vectorizer.vocabulary_)
print ("-"*120)
print ("Most Frequent 50 Words in the Document Collection:\n\n", fd.most_common(50))
print ("-"*120)

#Importing standard NLTK libraries for Stemming and Lemmatization
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

#Stemming and Lemmatizing
lmtzr = WordNetLemmatizer()
lemm_list=[]
porter_stemmer = PorterStemmer()
stemming_list=[]

for word in term_frequency_vectorizer.vocabulary_:
    stemming_list.append(porter_stemmer.stem(word))
    lemm_list.append(lmtzr.lemmatize(word))

print ("Stemming Unique Words List:\n\n",stemming_list)
print ("-"*120)
print ("Lemmatizing Unique Words List:\n\n",lemm_list)
print ("-"*120)

#Lemmatization and Position Tagging
from nltk import pos_tag
print ("Lemmatization and Position Tagging\n")
pos_tagging=[]
for document in document_collection:
    tokens = word_tokenize(document)
    tokens_pos = pos_tag(tokens)
    pos_tagging.append(tokens_pos)
print (pos_tagging)
print ("-"*120)

#Normalization of Document Collection
import re
print ("Normalized Document Collection\n\n")
for document in document_collection:
    
    for abbv in abb:
        document=document.replace(abbv, abb[abbv])

    #Lower Case
    document=document.lower()

    #Replace New Line Characters
    document=document.replace('\n',' ')

    #Remove Punctuation Marks
    punctuation_marks=['.', ',', '?', '!', '\'', '"', ':', ';', '...', '-']
    for sym in punctuation_marks:
        document=document.replace(sym, '')

    #Remove non alphanumeric characters
    document=re.sub(r'[^0-9a-zA-Z\s]+', '', document)

    #Numbers to Words
    numbers = [int(s) for s in document.split() if s.isdigit()]
    ie = inflect.engine()
    for n in numbers:
        document = document.replace(str(n), ie.number_to_words(n))
    print (document)
print ("-"*120)
