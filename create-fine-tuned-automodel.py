# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 13:06:29 2021

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
    
import pandas as pd

# Load the dataset into a pandas dataframe.
training_df1 = pd.read_csv("TheCartographicJournalLabeled2.csv", delimiter=';')
training_df2 = pd.read_csv("ALL_TGIS.csv", delimiter=',')
training_df3 = pd.read_csv("C_and_G.csv", delimiter=',')
training_df4 = pd.read_csv("IJGIS_con_abstract.csv", delimiter=',')

training_df = pd.concat([training_df1,training_df2,training_df3,training_df4])

validation_df = pd.read_csv("codigos_vs_articulos.csv", delimiter=',')

print('Number of training sentences: {:,}\n'.format(training_df.shape[0]))
print('Number of test sentences: {:,}\n'.format(validation_df.shape[0]))

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

from transformers import AutoTokenizer
from transformers import BertForMaskedLM

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
mlm = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

#mlmTrainingTensors = convertListOfSentencesToListOfTokensEmbeddings(abstractTrainingSentences[:10])

from transformers import AdamW


# activate training mode
mlm.train()
# initialize optimizer
optim = AdamW(mlm.parameters(), lr=5e-5)

def optimizeMlmModel(trainingSentences, percentageOfMaskedTokens):
  optim.zero_grad()
  inputs = None
  
  for sentence in trainingSentences:
    if (inputs is None):
      inputs = tokenizer.encode(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
    else:
      inputs_ids = tokenizer.encode(sentence, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
      inputs = torch.cat((inputs, inputs_ids), 0)

  masks = inputs.clone()
  # create random array of floats with equal dimensions to input_ids tensor
  rand = torch.rand(inputs.shape)
  # create mask array
  mask_arr = (rand < percentageOfMaskedTokens) * (inputs != 101) * \
            (inputs != 102) * (inputs != 0)

  selection = []

  for i in range(masks.shape[0]):
      selection.append(torch.flatten(mask_arr[i].nonzero()).tolist())
  
  for i in range(masks.shape[0]):
    masks[i, selection[i]] = 103
      
  outputs = mlm(masks,labels=inputs)
  loss = outputs.loss
  print(loss)
  loss.backward()
  # update parameters
  optim.step()
  print(loss.item())

cont = 0
while cont < 8600:
  optimizeMlmModel(abstractTrainingSentences[cont:cont+1], 0.2)
  cont+=1

mlm.save_pretrained('fine-tuned-mlm-as-pre-trained')