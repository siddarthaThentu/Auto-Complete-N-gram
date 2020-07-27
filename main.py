# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 20:05:25 2020

@author: siddarthaThentu
"""

import math
import random
import numpy as np
import pandas as pd
import nltk

with open("en_US.twitter.txt",encoding="utf8") as fp:
    data = fp.read()

#split the data read from the file into sentences
def split_to_sentences(data):
    
    sentences = [s.strip() for s in data.split("\n") if len(s)>0]
    
    return sentences

#tokenize the sentence into words
def tokenize_sentences(sentences):
    
    tokenized_sentences = [nltk.word_tokenize(sentence.lower()) for sentence in sentences]
    
    return tokenized_sentences

def get_tokenized_data(data):
    
    sentences = split_to_sentences(data)
    tokenized_sentences = tokenize_sentences(sentences)
    
    return tokenized_sentences

tokenized_data = get_tokenized_data(data)
random.seed(87)
random.shuffle(tokenized_data)

train_size = int(len(tokenized_data)*0.8)
train_data = tokenized_data[0:train_size]
test_data = tokenized_data[train_size:]

#print("{} data are split into {} train and {} test set".format(len(tokenized_data),len(train_data),len(test_data)))
'''        
print("First training sample:")
print(train_data[0])
print("First test sample")
print(test_data[0])
'''
def count_words(tokenized_sentences):
    
    word_counts = {}
    
    for sentence in tokenized_sentences:
        for token in sentence:
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1
                
    return word_counts

def get_words_with_nplus_frequency(tokenized_sentences,count_threshold):
    
    closed_vocab = []
    word_counts = count_words(tokenized_sentences)
    
    for word,count in word_counts.items():
        if(count>=count_threshold):
            closed_vocab.append(word)
    
    return closed_vocab

def replace_oov_words_by_unk(tokenized_sentences,vocabulary,unknown_token="<unk>"):
    
    vocabulary = set(vocabulary)
    
    replaced_tokenized_sentences = []
    
    for sentence in tokenized_sentences:
        replace_sentence = []
        for token in sentence:
            if token in vocabulary:
                replace_sentence.append(token)
            else:
                replace_sentence.append(unknown_token)
        replaced_tokenized_sentences.append(replace_sentence)
    
    return replaced_tokenized_sentences

def preprocess_data(train_data,test_data,count_threshold):
    
    vocabulary = get_words_with_nplus_frequency(train_data, count_threshold)
    train_data_replaced = replace_oov_words_by_unk(train_data, vocabulary)
    test_data_replaced = replace_oov_words_by_unk(test_data, vocabulary)
    
    return train_data_replaced,test_data_replaced,vocabulary

minimum_freq = 2
train_data_processed,test_data_processed,vocab = preprocess_data(train_data,test_data,minimum_freq)
    
def count_n_grams(data,n,start_token="<s>",end_token="<e>"):
    
    n_grams = {}
    
    for sentence in data:
        sentence = [start_token]*n + sentence + [end_token]
        sentence = tuple(sentence)
        for i in range(len(sentence)-n+1):
            n_gram = sentence[i:i+n]
            if n_gram not in n_grams:
                n_grams[n_gram] = 1
            else:
                n_grams[n_gram] += 1
    
    #print(n_grams)           
    return n_grams

def estimate_probability(word,previous_n_gram,n_gram_counts,
                         n_plus1_gram_counts,vocab_size,k=1.0):
    
    previous_n_gram = tuple(previous_n_gram)
    
    previous_n_gram_count = n_gram_counts.get((previous_n_gram),0)
    
    denominator = previous_n_gram_count + k*vocab_size
    
    n_plus1_gram = previous_n_gram + (word,)
    
    n_plus1_gram_count = n_plus1_gram_counts.get((n_plus1_gram),0)
    
    numerator = n_plus1_gram_count + k
    
    probability = numerator/denominator
    
    return probability

def estimate_probabilities(previous_n_gram,n_gram_counts,n_plus1_gram_counts,vocab,k=1.0):
    
    previous_n_gram = tuple(previous_n_gram)
    
    vocab = vocab + ["<e>","<unk>"]
    vocab_size = len(vocab)
    
    probabilities = {}
    
    for word in vocab:
        probability = estimate_probability(word,previous_n_gram,n_gram_counts,
                         n_plus1_gram_counts,vocab_size,k=k)
        probabilities[word] = probability
        
    return probabilities

def make_count_matrix(n_plus1_gram_counts,vocab):
    
    vocab = vocab + ["<e>","<unk>"]
    n_grams = []
    
    for n_plus1_gram in n_plus1_gram_counts.keys():
        n_gram = n_plus1_gram[0:-1]
        n_grams.append(n_gram)
    n_grams = list(set(n_grams))
    
    row_index = {n_gram:i for i,n_gram in enumerate(n_grams)}
    col_index = {word:j for j,word in enumerate(vocab)}
    
    count_matrix = np.zeros((len(n_grams),len(vocab)))
    
    for n_plus1_gram,count in n_plus1_gram_counts.items():
        n_gram = n_plus1_gram[:-1]
        word = n_plus1_gram[-1]
        if word not in vocab:
            continue
        i = row_index[n_gram]
        j = col_index[word]
        count_matrix[i,j] = count
        
    count_matrix = pd.DataFrame(count_matrix,index=n_grams,columns=vocab)
    
    return count_matrix

def make_probability_matrix(n_plus1_gram_counts,vocab,k):
    
    count_matrix = make_count_matrix(n_plus1_gram_counts,vocab)
    count_matrix += k
    prob_matrix = count_matrix.div(count_matrix.sum(axis=1),axis=0)
    
    return prob_matrix

def calculate_perplexity(sentence,n_gram_counts,n_plus1_gram_counts,vocab_size,k=1.0):
    
    n = len(list(n_gram_counts.keys())[0])
    sentence = ["<s>"]*n + sentence + ["<e>"]
    sentence = tuple(sentence)
    N = len(sentence)
    product_pi = 1.0
    
    for t in range(n,N):
        n_gram = sentence[t-n:t]
        word = sentence[t]
        probability = estimate_probability(word,n_gram,n_gram_counts,n_plus1_gram_counts,vocab_size)
        product_pi *= 1/probability
    
    perplexity = product_pi ** (1/N)
    
    return perplexity

def suggest_a_word(previous_tokens,n_gram_counts,n_plus1_gram_counts,vocabulary,k=1.0,start_with=None):
    
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = previous_tokens[-n:]
    probabilities = estimate_probabilities(previous_n_gram, n_gram_counts, n_plus1_gram_counts, vocabulary)
    suggestion=None
    max_prob = 0
    #print("\n",probabilities,"\n")
    for word,prob in probabilities.items():
        if start_with:
            if(not word.startswith(start_with)):
                continue
        if probabilities[word]>max_prob:
            suggestion = word
            max_prob = probabilities[word]
            
    return suggestion,max_prob

def get_suggestions(previous_tokens,n_gram_counts_list,vocab,k=1.0,start_with=None):
    
    model_counts = len(n_gram_counts_list)
    print("model_counts=",model_counts)
    suggestions = []
    for i in range(model_counts-1):
        n_gram_counts = n_gram_counts_list[i]
        n_plus1_gram_counts = n_gram_counts_list[i+1]
        
        suggestion = suggest_a_word(previous_tokens,n_gram_counts,n_plus1_gram_counts,
                                    vocab,k=k,start_with=start_with)
        suggestions.append(suggestion)
        
    return suggestions

inputSentence = input("Please enter a sentence : \n")
tok = get_tokenized_data(inputSentence)
print(tok)
n_gram_counts_list = []

for n in range(1,len(tok[0])+1):
    print("Computing n-gram counts with n =", n, "...")
    n_model_counts = count_n_grams(train_data_processed, n)
    n_gram_counts_list.append(n_model_counts)   

tmp_suggest = get_suggestions(tok[0],n_gram_counts_list,vocab,k=1.0)
print("\n",tmp_suggest,"\n")
'''
previous_tokens = ["i", "am", "to"]
tmp_suggest4 = get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest4)
    
previous_tokens = ["i", "want", "to", "go"]
tmp_suggest5 = get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest5)

previous_tokens = ["hey", "how", "are"]
tmp_suggest6 = get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest6)

previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest7 = get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0)

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest7)

previous_tokens = ["hey", "how", "are", "you"]
tmp_suggest8 = get_suggestions(previous_tokens, n_gram_counts_list, vocab, k=1.0, start_with="d")

print(f"The previous words are {previous_tokens}, the suggestions are:")
print(tmp_suggest8)
'''
    
        

        
    
    
    
    