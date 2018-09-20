#This code is entitled to Harsh Gandhi.
#The following code is for calculating the N-Gram probability using Laplace's Add-1 Smoothing

#Importing the NLTK library for python to perform some basic functions.
from nltk import ngrams
from nltk.tokenize import word_tokenize
import nltk

#BeautifulTable is a python library used to store the output in a tabular format. (Can be skipped)
from beautifultable import BeautifulTable
global table
table=BeautifulTable()

#Setting the column headers for the output table.
table.column_headers=["Gram","Probability"]

#Below is the corpu under examination.
str="""Natural language processing (NLP) is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data. Challenges in natural language processing frequently involve speech recognition, natural language understanding, and natural language generation. This is an example test sentence. Natural language processing (NLP) is the ability of a computer program to understand human language as it is spoken. NLP is a component of artificial intelligence (AI). The development of NLP applications is challenging because computers traditionally require humans to "speak" to them in a programming language that is precise, unambiguous and highly structured, or through a limited number of clearly enunciated voice commands. This is an example. Human speech, however, is not always precise -- it is often ambiguous and the linguistic structure can depend on many complex variables, including slang, regional dialects and social context. This is an example test sentence."""

#Here, is the test sentence whose probability of existence in the corpus is to be found.
sentence = 'This is an example test sentence.'

#Removing Punctuation Marks
punctuation_marks=['.', ',', '?', '!', '\'', '"', ':', ';', '...', '-','(',')']

#Removing punctuations from both the Corpus and the Test Sentence
for sym in punctuation_marks:
    str=str.replace(sym, '')
    sentence=sentence.replace(sym, '')

#Taking the user input for the value of N for N-Grams
n = input("Enter the value of N: ")
n=int(n)

#Total Number of N-Gram cases possible for the Corpus
no_cases=len(word_tokenize(str))-n+1

#Finding the Unique Words vocabulary set
words = nltk.tokenize.word_tokenize(str)
fdist1 = nltk.FreqDist(words)

filtered_word_freq = dict((word, freq) for word, freq in fdist1.items() if not word.isdigit())

#Forming the N-Gram for Corpus and the Test Sentence
test_sentence_grams = ngrams(word_tokenize(sentence),n)
test_sentence_grams1 = ngrams(word_tokenize(sentence),n-1)
corpus_grams = ngrams(word_tokenize(str),n)
corpus_grams1 = ngrams(word_tokenize(str),n-1)

#How Many N-1 Grams
c=0
for grams1 in corpus_grams1:
    c=c+1

#This is the list to store the individual gram-probability and shall be later used to calculate final probability.
prb_list=[]

#If Bigram
if n==2:
    V=len(filtered_word_freq)
    l=[]
    l1=[]
    for grams in corpus_grams:
        sub_sent=" ".join(grams)
        l.append(sub_sent)
    
    for grams in corpus_grams1:
        l1.append(grams[0])
    
    for grams in test_sentence_grams:
        sub_sent=" ".join(grams)
        count=l.count(sub_sent)
        count1=l1.count(sub_sent.split()[0])
        
        #The Laplace's Add One Smoothing formula for N-Grams
        probability=(count + 1)/(count1 + V)
        
        #Adding the Gram and it's probability to the table. Also appending the probability to the probability list.
        table.append_row([sub_sent,probability])
        prb_list.append(probability)

#For N>2 Grams
else:
    V=c
    
    #List to store N-Grams
    l=[]
    #List to store N-1 Grams
    l1=[]
    
    for grams in corpus_grams:
        sub_sent=" ".join(grams)
        l.append(sub_sent)
    
    for grams in corpus_grams1:
        sub_sent=" ".join(grams)
        l1.append(sub_sent)
    
    for grams in test_sentence_grams:
        sub_sent=" ".join(grams)
        ngram=sub_sent
        count=l.count(sub_sent)
        
        sub_sent=sub_sent.split()
        
        #Removing the last element to get N-1 gram
        sub_sent.pop()
        sub_sent1=" ".join(sub_sent)
        count1=l1.count(sub_sent1)

        #Probability formula of Laplace-Add 1 Smoothing
        probability=(count + 1)/(count1 + V)
        
        #Adding the Gram and it's probability to the table. Also appending the probability to the probability list.
        table.append_row([ngram,probability])
        prb_list.append(probability)

print (table)
Fin_probability=1

#Calculating the Final Probability from the individual probabilities.
for prob in prb_list:
    Fin_probability=Fin_probability*prob
print ("\nFinal Probability: ", Fin_probability)
