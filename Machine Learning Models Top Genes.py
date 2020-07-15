#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:38:58 2019

@author: xander
"""

import copy, matplotlib.pyplot as plt, os, pandas as pd, sklearn
os.path.join(os.path.dirname(__file__))

#Open file
gene_Frame = pd.read_csv('topgenes.csv')
features = gene_Frame.columns

#Seperate categorical and non-categorical data and normalize it
category_Data = gene_Frame[features[0:4]]
norm_Data = gene_Frame[features[4:]]
norm_Data = norm_Data / norm_Data.max()

#Noncategorical data
'''category_Data = gene_Frame[features[0]]
norm_Data = gene_Frame[features[1:]]
norm_Data = norm_Data / norm_Data.max()'''

#Combine categorical and non categorical data into new dataframe
data_Frame = norm_Data
#Used for Categorical data
data_Frame[features[0:4]] = category_Data
#Used for Noncategorical data
#data_Frame[features[0]] = category_Data
data_Frame = data_Frame[features]

#Shuffle the dataframe
sklearn.utils.shuffle(data_Frame)

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
    #Best scores
    #[0.8529411764705882, 0.7941176470588235, 0.7941176470588235, 0.7352941176470589, 0.7647058823529411, 0.6470588235294118, 0.6764705882352942, 0.6470588235294118, 0.6470588235294118, 0.6470588235294118, 0.6470588235294118]
    #Neighbor Count for best scores
    #[3, 4, 5, 5, 7, 3, 5, 7, 4, 4, 7]
    #Best scores Non-categorical
    #[0.8235294117647058, 0.7941176470588235, 0.7647058823529411, 0.7647058823529411, 0.7352941176470589, 0.6764705882352942, 0.6764705882352942, 0.6764705882352942, 0.7058823529411765, 0.6176470588235294, 0.6764705882352942]
    #Neighbor Count for best scores
    #[3, 3, 3, 3, 25, 3, 3, 3, 3, 3, 3]
    '''
    from sklearn import neighbors
    for j in range (2,len(train_Data)):
        nbrs = neighbors.KNeighborsClassifier(n_neighbors = j, weights = 'uniform')
        nbrs.fit(train_Data,train_Label)
        models[num_Groups].append(nbrs)
        scores[num_Groups].append(nbrs.score(test_Data, test_Label))
    '''
    #Best scores Categorical
    #[0.6470588235294118, 0.5, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.5294117647058824, 0.4117647058823529, 0.47058823529411764, 0.47058823529411764, 0.47058823529411764, 0.5294117647058824]
    #Best scores Non-categorical
    #[0.7058823529411765, 0.5588235294117647, 0.5882352941176471, 0.4411764705882353, 0.47058823529411764, 0.5, 0.4411764705882353, 0.35294117647058826, 0.5294117647058824, 0.5294117647058824, 0.5882352941176471]
    
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_Data,train_Label)
    models[num_Groups].append(clf)
    scores[num_Groups].append(clf.score(test_Data, test_Label))
    
    #Best scores
    #[0.7352941176470589, 0.7352941176470589, 0.7058823529411765, 0.6764705882352942, 0.7352941176470589, 0.7352941176470589, 0.6176470588235294, 0.6764705882352942, 0.6470588235294118, 0.6470588235294118, 0.6176470588235294]
    #Number of Trees
    #[80, 20, 40, 10, 10, 90, 10, 60, 10, 80, 10]
    #Best scores Non-categorical
    #[0.7058823529411765, 0.6470588235294118, 0.7058823529411765, 0.6764705882352942, 0.6470588235294118, 0.6470588235294118, 0.6176470588235294, 0.6764705882352942, 0.6764705882352942, 0.6176470588235294, 0.6176470588235294]
    #Number of Trees
    #[10, 50, 30, 70, 60, 20, 10, 20, 50, 40, 30]
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

#Feature Weights for Decision Tree
'''imp_Features = []
for i in best_Models:
    imp_Features.append(i.feature_importances_)
feature_Frame = pd.DataFrame(imp_Features, columns = features[1:], index = range(4,15))'''

#Create ROC curve for the best model for each number of classes
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
    #plt.savefig('ROC_for_KNN_Top_Genes_Non_Categorical_' + str(i + 4) + '_Classes.png')
    #For Decision Tree
    #plt.savefig('ROC_for_DecisionTree_Top_Genes_Non_Categorical_' + str(i + 4) + '_Classes.png')
    #For RandomForest
    #plt.savefig('ROC_for_RandomForest_Top_Genes_Non_Categorical_' + str(i + 4) + '_Classes.png')