# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 10:59:27 2021

@author: Lucas
"""

import numpy as np

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

max_length_characters = 500
abstractTrainingSentences = training_df.loc[ (training_df.Abstract != '[No abstract available]') & (training_df.Abstract.str.replace(',','').str.split().str.len() < max_length_characters) ].Abstract.values
titleTrainingSentences = training_df.loc[ (training_df.Title != '[No abstract available]') & (training_df.Title.str.replace(',','').str.split().str.len() < max_length_characters) ].Title.values
trainingSentences = np.concatenate((abstractTrainingSentences, titleTrainingSentences), axis=None)

labels = [1 for index in trainingSentences]
trainingLabels = np.array(labels)

abstractTestSentences = validation_df.loc[(validation_df.Abstract.str.replace(',','').str.split().str.len() < 300) & (validation_df.votos_CIG != validation_df.votos_No_CIG) & (validation_df.votos_CIG.isnull() == False) ]
titleTestSentences = validation_df.loc[(validation_df.Title.str.replace(',','').str.split().str.len() < 300) & (validation_df.votos_CIG != validation_df.votos_No_CIG) & (validation_df.votos_CIG.isnull() == False) ]

testOrdinals = abstractTestSentences.Ordinal.apply(int).values


predictionsOfBestAbstractClassification = [1,1,-1,-1,1,1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,1,1,1,1
,-1,1,1,1,-1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,1,1,-1,-1,1,1,1,-1
,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1
,1,-1,1,1,-1,-1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,1,-1
,-1,1,-1,-1,1,1,-1,-1,1,1,1,1,-1,-1,1,-1,1,1,-1,-1,1,-1,1,-1
,-1,1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1
,1,-1,-1,1,-1,-1,-1,-1,1,1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,1,-1,-1
,-1,-1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,-1,1,1]

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