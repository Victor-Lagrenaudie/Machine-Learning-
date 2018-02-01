# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:53:50 2017
@author: Abhijeet Singh
"""


import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix

def make_y_test(main_dir):
    total_line = return_number_of_line(main_dir)
    half_line = int(total_line / 2)
    y_test = np.empty(half_line)
    with open(main_dir) as fi:
            for (i , line) in enumerate(fi):
                     if (i >=(half_line)):
                         words = line.split()
                         for word in words:
                             if word =='ham':
                                 y_test[i-half_line]=0

                             elif word =='spam':
                                 y_test[i-half_line] = 1



            return y_test

def make_y_train(main_dir):
    total_line = return_number_of_line(main_dir)
    half_line = int(total_line / 2)

    y_train = np.empty(half_line)

    with open(main_dir) as fi:
            for (i , line) in enumerate(fi):
                     if (i <(half_line)):
                         words = line.split()
                         for word in words:
                             if word =='ham':
                                 y_train[i]=0
                             elif word =='spam':
                                 y_train[i] = 1

            return y_train

def return_number_of_line(main_dir):
    with open(main_dir) as f:
        line_count = 0
        for line in f:
            line_count += 1
    return line_count


def make_Dictionary(train_dir):  ##Creation of the dictionnary
    y_train = np.empty(half_line)
    all_words = []   #future matrice of words

    with open(main_dir) as fi:             ##open directory
            for (i , line) in enumerate(fi):
                     if (i <(half_line)):
                         words = line.split()
                         for word in words:
                             if ( (word !='ham') and (word != 'spam')):
                                 all_words.append(word)


            dictionary = Counter(all_words)
            for item in list(dictionary):

                if item.isalpha() == False:
                    del dictionary[item]
                elif len(item) == 1:
                    del dictionary[item]
            dictionary = dictionary.most_common(3000)

            return dictionary

def extract_features(dir,type):

    total_line = return_number_of_line(main_dir)
    half_line = int(total_line / 2)
    features_matrix = np.zeros((half_line, 3000))

    lineID = 0

    with open(dir) as fi:
        for i, line in enumerate(fi):
            if(type=='train'):
                if(i<half_line):
                    words = line.split()
                    for word in words:
                        wordID = 0
                        for i, d in enumerate(dictionary):
                            if d[0] == word:
                                wordID = i
                                features_matrix[lineID, wordID] = words.count(word)
                    lineID = lineID + 1

            elif(type=='test'):
                if (i >= half_line):
                    words = line.split()
                    for word in words:
                        if ((word != 'ham') and (word != 'spam')):

                            wordID = 0
                            for i, d in enumerate(dictionary):
                                if d[0] == word:
                                    wordID = i
                                    features_matrix[lineID, wordID] = words.count(word)
                    lineID = lineID + 1

    return features_matrix



main_dir = 'messages.txt'
total_line = return_number_of_line(main_dir)
half_line = int(total_line / 2)

dictionary=make_Dictionary(main_dir)
y_train=make_y_train(main_dir)
train_matrix = extract_features(main_dir,'train')

y_test=make_y_test(main_dir)
test_matrix = extract_features(main_dir,'test')

# Training SVM and Naive bayes classifier and its variants

model1 = LinearSVC()
model2 = MultinomialNB()

model1.fit(train_matrix, y_train)
model2.fit(train_matrix, y_train)

result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)

print("score model 1: %s"% model1.score(test_matrix,y_test))
print("score model 2: %s"% model2.score(test_matrix,y_test))
confusion_matrix_1=confusion_matrix(y_test, result1)

print("confusion matrix model 1: ")
print(confusion_matrix_1)
confusion_matrix_2=confusion_matrix(y_test, result2)
print("confusion matrix model 2:")
print(confusion_matrix_2)

precision_matrix_1=confusion_matrix_1[0,0]/(confusion_matrix_1[0,0] + confusion_matrix_1[0,1])
recall_matrix_1=confusion_matrix_1[0,0]/(confusion_matrix_1[0,0] + confusion_matrix_1[1,0])
F_matrix_1=2*(precision_matrix_1*recall_matrix_1)/(precision_matrix_1+recall_matrix_1)
print("F score model 1: %s"% F_matrix_1)

precision_matrix_2=confusion_matrix_2[0,0]/(confusion_matrix_2[0,0] + confusion_matrix_2[0,1])
recall_matrix_2=confusion_matrix_2[0,0]/(confusion_matrix_2[0,0] + confusion_matrix_2[1,0])
F_matrix_2=2*(precision_matrix_2*recall_matrix_2)/(precision_matrix_2+recall_matrix_2)


print("F score model 2: %s"% F_matrix_2)
