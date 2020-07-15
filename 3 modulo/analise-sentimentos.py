#!/usr/bin/env python
# coding: utf-8

# In[1]:

import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.layers import Flatten
import copy

palavras_mais_usadas = 10000
tamanho_maximo = 500
tamanho_da_camada_de_incorporacao = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=palavras_mais_usadas)

x_train = sequence.pad_sequences(x_train,tamanho_maximo)
x_test = sequence.pad_sequences(x_test,tamanho_maximo)


# In[2]:


json_dicionario = imdb.get_word_index()
dicionarioPalavras = list(json_dicionario.keys())
dicionarioNumeros = list(json_dicionario.values())
def traduzir(revisao):
    retorno = []
    for palavra in revisao:
        if(palavra == 0):
            retorno.append('')
        elif(palavra == 1):
            retorno.append('>')
        elif(palavra == 2):
            retorno.append('?')
        else:
            index = dicionarioNumeros.index(palavra - 3)
            retorno.append(dicionarioPalavras[index])
    return ' '.join(retorno)  


# In[3]:


model = Sequential()
model.add(Embedding(palavras_mais_usadas, tamanho_da_camada_de_incorporacao, input_length = tamanho_maximo))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
print(model.summary())


# In[ ]:


treino = model.fit(x_train,y_train,validation_data = (x_test,y_test), epochs=3, batch_size=128 )


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
get_ipython().magic(u'matplotlib inline')
precisao = treino.history['accuracy']
validacao = treino.history['val_accuracy']
epochs = range(1, len(precisao) + 1)
plt.plot(epochs, precisao, '-', label='Precisão do treinamento')
plt.plot(epochs, validacao, ':', label='Precisão da validação')
plt.title('Precisão do treinamento e da validação')
plt.xlabel('Época')
plt.ylabel('Precisão')
plt.legend(loc='upper left')
plt.plot()


# In[ ]:


scores = model.evaluate(x_test,y_test,verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


# In[ ]:


import string
import numpy as np
def analise(texto):
    tabela = str.maketrans("","", string.punctuation)
    traducao = texto.translate(tabela)
    traducao = traducao.lower().split(" ")
    traducao = [palavra for palavra in traducao if palavra.isalpha()]

    
    critica = []
    critica.append(1)
    for palavra in traducao:
        if palavra in json_dicionario and json_dicionario[palavra] <= palavras_mais_usadas:
            critica.append(json_dicionario[palavra])
        else:
            critica.append(2)
    padded_input = sequence.pad_sequences([critica], maxlen=tamanho_maximo)
    result = model.predict(np.array([padded_input][0]))[0][0]
    return result
print(analise("this movie was wonderful."))
print(analise('Easily the most stellar experience I have ever had.'))
print(analise('The long lines and poor customer service really turned me off.'))


# In[ ]:




