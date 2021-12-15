# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 17:20:04 2021
@author: Lucas
"""

import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')
    
########################################################################
    
import torch

# If there's a GPU available...
if torch.cuda.is_available():    

    # Tell PyTorch to use the GPU.    
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

########################################################################

## If server complains about transformer dependence
##!pip install transformers

import pandas as pd

# Load the dataset into a pandas dataframe.
training_df1 = pd.read_csv("TheCartographicJournalLabeled2.csv", delimiter=';')
training_df2 = pd.read_csv("ALL_TGIS.csv", delimiter=',')
training_df3 = pd.read_csv("C_and_G.csv", delimiter=',')
training_df4 = pd.read_csv("IJGIS_con_abstract.csv", delimiter=',')

training_df = pd.concat([training_df1,training_df2,training_df3,training_df4])

# Report the number of sentences.

validation_df = pd.read_csv("codigos_vs_articulos.csv", delimiter=',')

print('Number of training sentences: {:,}\n'.format(training_df.shape[0]))
print('Number of test sentences: {:,}\n'.format(validation_df.shape[0]))

########################################################################
## Filter training and test corpus for sentences with less than 250 tokens.
########################################################################

# Get the lists of sentences and their labels.
import numpy as np

max_length_characters = 500
abstractTrainingSentences = training_df.loc[ (training_df.Abstract != '[No abstract available]') & (training_df.Abstract.str.replace(',','').str.split().str.len() < max_length_characters) ].Abstract.values
titleTrainingSentences = training_df.loc[ (training_df.Title != '[No abstract available]') & (training_df.Title.str.replace(',','').str.split().str.len() < max_length_characters) ].Title.values
trainingSentences = np.concatenate((abstractTrainingSentences, titleTrainingSentences), axis=None)

labels = [1 for index in trainingSentences]
trainingLabels = np.array(labels)

abstractTestSentences = validation_df.loc[(validation_df.Abstract.str.replace(',','').str.split().str.len() < 300) & (validation_df.votos_CIG != validation_df.votos_No_CIG) & (validation_df.votos_CIG.isnull() == False) ]
titleTestSentences = validation_df.loc[(validation_df.Title.str.replace(',','').str.split().str.len() < 300) & (validation_df.votos_CIG != validation_df.votos_No_CIG) & (validation_df.votos_CIG.isnull() == False) ]

testOrdinals = abstractTestSentences.Ordinal.apply(int).values

abstractTestLabels = []
for index, row in abstractTestSentences.iterrows():
  if int(row['votos_CIG']) < int(row['votos_No_CIG']):
      abstractTestLabels.append(-1)
  else:
      abstractTestLabels.append(1)

titleTestLabels = []
for index, row in titleTestSentences.iterrows():
  if int(row['votos_CIG']) < int(row['votos_No_CIG']):
      titleTestLabels.append(-1)
  else:
      titleTestLabels.append(1)

titleTestSentences = titleTestSentences.Title.values
abstractTestSentences = abstractTestSentences.Abstract.values

print("Size of titles train set: ",len(titleTrainingSentences))
print("Size of titles test set: ",len(titleTestSentences),"\n")

print("Size of abstract train set: ",len(abstractTrainingSentences))
print("Size of abstract test set: ",len(abstractTestSentences),"\n")

print("Size of abstract+title train set: ",len(trainingSentences))

########################################################################
import torch
import tensorflow as tf
import time
from transformers import AutoModel
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
model = AutoModel.from_pretrained('bert-large-cased')

tokenizations_time_metrics = []

## Aqui ele pega o texto original e gera os tokens
def convertSentenceToBERTEmbedding(sentence):
  try:
    # input_ids = torch.tensor([tokenizer.encode(sentence).ids])
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    with torch.no_grad():
        start_time = time.time()
        outs = model(input_ids)
        encoded = outs[0][0, 1:-1]  # Ignore [CLS] and [SEP] special tokens

        #Aqui faz a média
        out = tf.reduce_mean(encoded,0)
        tokenizations_time_metrics.append([len(encoded),(time.time() - start_time)])
    return out
  except :
    print("bigger than 512 tokens")
    return None

########################################################################

def convertListOfSentencesToListOfTokensEmbeddings(list_of_sentences):
  listOfEmbeddings = []

  i = 0;
  for sentence in list_of_sentences:
    print(i)
    sentenceEmbedding = convertSentenceToBERTEmbedding(sentence)
    if( sentenceEmbedding != None):
      listOfEmbeddings.append(sentenceEmbedding)
    i+=1

  return listOfEmbeddings


abstractTestEmbeddings = convertListOfSentencesToListOfTokensEmbeddings(abstractTestSentences)
#abstractTrainingEmbeddings = convertListOfSentencesToListOfTokensEmbeddings(abstractTrainingSentences)

titleTestEmbeddings = convertListOfSentencesToListOfTokensEmbeddings(titleTestSentences)
#titleTrainingEmbeddings = convertListOfSentencesToListOfTokensEmbeddings(titleTrainingSentences)

trainingEmbeddings = convertListOfSentencesToListOfTokensEmbeddings(trainingSentences)

#print("Number of abstract training embeddings: ",len(abstractTrainingEmbeddings))
print("Number of abstract test embeddings: ",len(abstractTestEmbeddings))
print("Number of title test embeddings: ",len(titleTestEmbeddings))
print("Number of training embeddings: ",len(trainingEmbeddings))

########################################################################

#filter max training size for a constant of 18000 instances
indices = range(0,18000)

trainingEmbeddingsFiltered = []
trainingLabelsFiltered = []

for i in indices:
  trainingEmbeddingsFiltered.append(trainingEmbeddings[i])
  trainingLabelsFiltered.append(trainingLabels[i])

print(len(trainingEmbeddingsFiltered))

########################################################################
## Find hyperparameters for SVM One Class
########################################################################
from sklearn import svm
from sklearn.model_selection import ParameterGrid
from sklearn import metrics

svm_time_metrics = []
start_time = time.time()

def fineTuneSVMOneClass(gammas):
  bestAbstractAccuracy = 0
  bestAbstractRecall = []
  bestAbstractParameter = None
  predictionsOfBestAbstractClassification = []

  bestTitleAccuracy = 0
  bestTitleRecall = []
  bestTitleParameter = None
  predictionsOfBestTitleClassification = []

  def classifyAndPrintMetricsInTest(validation_input,validation_labels,test_description):
      predicted_labels = tuned_ocsvm.predict(validation_input)
      print(predicted_labels)
      metrics = getMetrics(validation_labels, predicted_labels)

      print(test_description," accuracy: ",metrics[0]," recall: ",metrics[1])
      return predicted_labels

  def getMetrics(validation_labels, predicted_labels):
    accuracy = metrics.accuracy_score(validation_labels, predicted_labels, normalize=True, sample_weight=None)
    recall = metrics.recall_score(validation_labels, predicted_labels, labels=None, pos_label=1, average=None, sample_weight=None, zero_division='warn')

    return [accuracy,recall]

  tuned_ocsvm = svm.OneClassSVM()
  nus = [0.1]
  scores = ['recall']
  tuned_parameters = {'kernel' : ['rbf'], 'gamma' : gammas, 'nu': nus}

  for z in ParameterGrid(tuned_parameters):
      print(z)
      tuned_ocsvm.set_params(**z)
      tuned_ocsvm.fit(trainingEmbeddingsFiltered, trainingLabelsFiltered)

      abstractPredictedLabels = classifyAndPrintMetricsInTest(abstractTestEmbeddings, abstractTestLabels, "Classifying abstracts: ")
      abstractMetrics = getMetrics(abstractTestLabels, abstractPredictedLabels)
      abstractAccuracy = abstractMetrics[0]
      abstractRecall = abstractMetrics[1]

      if(abstractAccuracy > bestAbstractAccuracy):
        bestAbstractAccuracy = abstractAccuracy
        bestAbstractRecall = abstractRecall
        bestAbstractParameter = z
        predictionsOfBestAbstractClassification = abstractPredictedLabels

      titlePredictedLabels = classifyAndPrintMetricsInTest(titleTestEmbeddings, titleTestLabels, "Classifying titles: ")
      titleMetrics = getMetrics(titleTestLabels, titlePredictedLabels)
      titleAccuracy = titleMetrics[0]
      titleRecall = titleMetrics[1]

      if(titleAccuracy > bestTitleAccuracy):
        bestTitleAccuracy = titleAccuracy
        bestTitleRecall = titleRecall
        bestTitleParameter = z
        predictionsOfBestTitleClassification = titlePredictedLabels
      
  print("\n")
  print("Best found accuracy for abstracts: ",bestAbstractAccuracy," Best recall ",bestAbstractRecall," for parameters ",bestAbstractParameter)
  print("\n")
  print("Best found accuracy for titles: ",bestTitleAccuracy," Best recall ",bestTitleRecall," for parameters ",bestTitleParameter)
  return [bestAbstractAccuracy, bestAbstractRecall, bestAbstractParameter, predictionsOfBestAbstractClassification]

gammas = [0.01,0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
classifications = fineTuneSVMOneClass(gammas)
predictionsOfBestAbstractClassification = classifications[3]
bestParameters = classifications[2]
bestGamma = bestParameters['gamma']
gammas1 = [bestGamma - (i/100) for i in range(1,5)  if (bestGamma - (i/100)) > 0]
gammas2 = [bestGamma + (i/100) for i in range(1,5)]
gammas = gammas1 + [bestGamma] + gammas2
print(bestGamma,' ',gammas)
classifications = fineTuneSVMOneClass(gammas)
predictionsOfBestAbstractClassification = classifications[3]
bestParameters = classifications[2]
bestGamma = bestParameters['gamma']

svm_time_metrics.append([len(trainingEmbeddings),(time.time() - start_time)])

############################################################################################################
###### Merge classification results with indexes of test articles
############################################################################################################


bert_etiquetado_df = pd.DataFrame({'Documents': testOrdinals, '13': predictionsOfBestAbstractClassification})
bert_etiquetado_df["13"].replace({-1: 0}, inplace=True)
concentrado = pd.read_csv(r"concentrado_etiquetadores.csv") # Aqui leo  los datos

bert_etiquetado_merged = pd.merge(concentrado,bert_etiquetado_df,on="Documents", how = 'left')
bert_etiquetado_df = bert_etiquetado_merged[["Documents","13"]]

#############################################################################################################
##### Calculate alphs
#############################################################################################################

import krippendorff
import random

concentrado.drop(['Documents'], axis=1, inplace=True) # Borro una columna del archivo
bert_etiquetado_df.drop(['Documents'], axis=1, inplace=True) # Borro una columna del archivo

def calculoKrippendorff(reliability_data_str):
    reliability_data = reliability_data_str
    return(krippendorff.alpha(reliability_data=reliability_data, level_of_measurement='nominal')) #nominal, interval, ratio

# Luego para hacer el cálculo hago iteraciones:

columnas = ['1','2','3','4','5','6','7','8','9','10','11','12']
z = []

for i in columnas:
    print('\n')
    print('reemplazó la columna '+str(i)+' por la maquina, columna 13')
    
    aaa = concentrado.drop(columns=i)
    bert_judment_column = bert_etiquetado_df["13"]
    aaa = aaa.join(bert_judment_column)
    
    for j in aaa:
        
        ##La columna de nombre 13 contiene informacion de bert
        if(j == '13'):
            continue;
        print('borró la columna ',j)
            
        aaa = aaa.drop(columns= j)
        print(aaa.columns)
        z.append(calculoKrippendorff(aaa.transpose()))
        
        aaa = concentrado.drop(columns=i)
        bert_judment_column = bert_etiquetado_df["13"]
        aaa = aaa.join(bert_judment_column)

print("Máximo",np.max(z),"Mínimo", np.min(z), "Promedio",np.mean(z))