#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:38:58 2019

@author: xander
"""

import copy, matplotlib.pyplot as plt, os, pandas as pd, sklearn
os.path.join(os.path.dirname(__file__))

#Open file
blood_Data = []
with open ('Blood_SignifGeneTPMs_Covariates.txt') as file:
    text = file.readlines()
    for line in text:
        line = line.strip().split('\t')
        blood_Data.append(line)        
file.close()

#Create data frame from file
features = blood_Data.pop(0)
data_Frame = pd.DataFrame(blood_Data, columns = features)
data_Frame = data_Frame.astype('float')

#Shuffle the dataframe
sklearn.utils.shuffle(data_Frame)

#Normalizing all Data
#norm_Data = data_Frame / data_Frame.max()

#Seperate categorical and non-categorical data and normalize it
norm_Data = data_Frame[features[3:]]
norm_Data = norm_Data / norm_Data.max()
category_Data = data_Frame[features[0:3]]

#Combine categorical and non categorical data into new dataframe
data_Frame = norm_Data
data_Frame[features[0:3]] = category_Data
data_Frame = data_Frame[features]

#Get the PMI from the dataframe and convert to hours
labels = data_Frame.pop(features[0])
labels = labels/60
norm_Labels = [0] * len(labels)

#Seperate data into training and testing data
cutoff = int(data_Frame.shape[0]*0.8)
train_Data = data_Frame[0:cutoff]
test_Data = data_Frame[cutoff:]

#Divide PMI between 0 and 12 hours into i evenly spaced intervals
#Create classification models using the i groups
models = {}
scores = {}
test_Labels = {}
for i in range (2,13):
    num_Groups = i + 2
    models[num_Groups] = []
    scores[num_Groups] = []
    for j in range (0,len(labels)):
        for k in range (1, num_Groups):
            if (labels[j] == 0):
                norm_Labels[j] = 0
                break
            elif ((12/i)*(k-1) <= labels[j]) and (labels[j] < (12/i)*k):
                norm_Labels[j] = k
                break
            elif (labels[j] >= 12):
                norm_Labels[j] = num_Groups - 1
                break
    train_Label = norm_Labels[0:cutoff]
    test_Label = norm_Labels[cutoff:]
    test_Labels[i] = test_Label
    #All Normalized Data best scores
    #[0.6176470588235294, 0.5588235294117647, 0.5588235294117647, 0.5588235294117647, 0.5588235294117647, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824]
    #Neighbor Count for best scores
    #[23, 25, 23, 22, 22, 20, 20, 20, 20, 18, 18]
    #With Categorical Data best scores
    #[0.6764705882352942, 0.6470588235294118, 0.6764705882352942, 0.6176470588235294, 0.6764705882352942, 0.6470588235294118, 0.6176470588235294, 0.6176470588235294, 0.6176470588235294, 0.6176470588235294, 0.6176470588235294]
    #Neighbor Count for best scores
    #[16, 16, 43, 16, 21, 23, 12, 22, 13, 13, 12]
    '''
    from sklearn import neighbors
    for j in range (2,len(train_Data)):
        nbrs = neighbors.KNeighborsClassifier(n_neighbors = j, weights = 'uniform')
        nbrs.fit(train_Data,train_Label)
        models[num_Groups].append(nbrs)
        scores[num_Groups].append(nbrs.score(test_Data, test_Label))
    '''
    #All Normalized
    #[0.5882352941176471, 0.4411764705882353, 0.4411764705882353, 0.4411764705882353, 0.3235294117647059, 0.4411764705882353, 0.38235294117647056, 0.4117647058823529, 0.35294117647058826, 0.3235294117647059, 0.3235294117647059]
    #With Categorical Data
    #[0.5294117647058824, 0.4117647058823529, 0.4411764705882353, 0.38235294117647056, 0.3235294117647059, 0.47058823529411764, 0.4117647058823529, 0.38235294117647056, 0.38235294117647056, 0.38235294117647056, 0.35294117647058826]
    '''
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_Data,train_Label)
    models[num_Groups].append(clf)
    scores[num_Groups].append(clf.score(test_Data, test_Label))
    '''
    #All Normalized
    #[0.7352941176470589, 0.6764705882352942, 0.6470588235294118, 0.7058823529411765, 0.5588235294117647, 0.5882352941176471, 0.6764705882352942, 0.5294117647058824, 0.5588235294117647, 0.6176470588235294, 0.5882352941176471]
    #Number of Trees
    #[10, 90, 20, 50, 10, 40, 90, 20, 70, 80, 30]
    #With Categorical Data
    #[0.6470588235294118, 0.5882352941176471, 0.5882352941176471, 0.6176470588235294, 0.6176470588235294, 0.6176470588235294, 0.6176470588235294, 0.5294117647058824, 0.5882352941176471, 0.5588235294117647, 0.5882352941176471]
    #Number of Trees
    #[50, 90, 30, 30, 60, 90, 90, 60, 60, 20, 20]
    '''
    from sklearn import ensemble
    for j in range (10, 100, 10):
        clf = ensemble.RandomForestClassifier(n_estimators = i)
        clf.fit(train_Data, train_Label)
        models[num_Groups].append(clf)
        scores[num_Groups].append(clf.score(test_Data, test_Label))
    '''

#Determine most accurate model for each number of classes and get model parameters
best_Scores = []
best_Models = []
model_Cnts = []
for i in scores.keys():
    best_Scores.append(max(scores[i]))
    best_Models.append(models[i][scores[i].index(best_Scores[-1])])
    #Evaluating KNN K amount
    #model_Cnts.append(scores[i].index(best_Scores[-1]) + 2)
    #Evaluating Tree Amount
    #model_Cnts.append((scores[i].index(best_Scores[-1]) + 1) * 10)

#Create ROC curve for the best model for each number of classes
'''
for i in range (0,len(best_Models)):
    test_Probs = best_Models[i].predict_proba(test_Data)
    fig = plt.figure()
    plt.set_cmap('tab20')
    ax = plt.subplot(111)
    class_Label = copy.deepcopy(test_Labels[i + 2])
    mod_Classes = list(set(class_Label))
    for j in mod_Classes:
        class_Probs = test_Probs[:,mod_Classes.index(j)]
        class_Label = copy.deepcopy(test_Labels[i + 2])
        fpr,tpr,thresholds = sklearn.metrics.roc_curve(class_Label,class_Probs,j)
        ax.plot(fpr, tpr, label=('Group ' + str(j)))
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width*0.8, chartBox.height])
    ax.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), shadow=True, ncol=1)
    #For KNN
    #plt.savefig('ROC_for_KNN_All_Features_Categorical_' + str(i + 4) + '_Classes.png')
    #plt.savefig('ROC_for_KNN_All_Features_Normative_' + str(i + 4) + '_Classes.png')
    #For Decision Tree
    #plt.savefig('ROC_for_DecisionTree_All_Features_Categorical_' + str(i + 4) + '_Classes.png')
    #plt.savefig('ROC_for_DecisionTree_All_Features_Normative_' + str(i + 4) + '_Classes.png')
    #For RandomForest
    #plt.savefig('ROC_for_RandomForest_All_Features_Categorical_' + str(i + 4) + '_Classes.png')
    #plt.savefig('ROC_for_RandomForest_All_Features_Normative_' + str(i + 4) + '_Classes.png')
'''