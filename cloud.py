# %%
from pathlib import Path
import scipy.io
from glob import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import pandas as pd
from sklearn.preprocessing import normalize, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score

# %% [markdown]
# # 1. Aquisição dos Dados

# %%
data = Path('data')

signals = []
classes =[]

for value in sorted(data.glob('*.mat')):
  signal = scipy.io.loadmat(value)
  signal = np.array(signal["val"], dtype='float')[0]
  signals.append(signal)
  # print(value)
  
classification_path = os.path.join(data, "classification.txt")
with open(classification_path) as f:
    classes = f.readlines()

for i in range(0, len(classes)):
  classes[i] = classes[i][classes[i].find(','):]
  classes[i] = classes[i][: classes[i].find('\n')]
  classes[i] = classes[i].replace(",", "")

# %% [markdown]
# # 2. Preprocessamento

# %% [markdown]
# ### 2.1 Análise da distribuição das classes

# %%
d = {'signals': signals, 'classes': classes}
df = pd.DataFrame(data=d)
a = df.groupby('classes')
for i in ['N', 'O', 'A', '~']:
  print(i)
  aux = []

  for j in a.get_group(i)["signals"]:
    aux.append(len(j))

  print('Média:', round(np.mean(aux), 2))
  print("Std:", round(np.std(aux), 2))
  print("Qntd:", len(aux))
  print("Min:", min(aux))
  print("---------------")

# %%
plt.rcParams.update({'font.size': 20})

df["classes"].hist(
    figsize=(8, 6),
    grid = False,
    bins = 10,
)
plt.xticks(range(4))
plt.show()

# %% [markdown]
# ### 2.2 Wavelet helping functions

# %%
def gera_filtro_MH(Escala):
  passo = 1/Escala
  X = []
  a = -5
  X.append(a)
  while a <= 5:
    a=a+passo
    X.append(a)
  Y = []
  for i in range(0,len(X)):
    Y.append(2.1741*(1/math.sqrt(2*math.pi) * (1 - X[i]**2) * np.exp(-X[i]**2/(2))))
  return Y

def apply_wavelet(signal, level):
  Filtro = gera_filtro_MH(level)
  y = np.convolve(signal,Filtro)
  return y

# %% [markdown]
# ### 2.3 Aplicação da tranformada Wavelet

# %%
plt.rcParams['figure.figsize']=(20,5)

signal = df[df["classes"] == '~'].iloc[0]["signals"][0:500]
fig, ax = plt.subplots(2,1)
fig.suptitle("Too Noisy for analyzes (~)")
z = apply_wavelet(signal, 2**2)
ax[0].plot(signal)
ax[1].plot(z, 'r')
plt.show()

signal = df[df["classes"] == 'N'].iloc[0]["signals"][0:500]
fig, ax = plt.subplots(2,1)
fig.suptitle("Sinusal (N)")
z = apply_wavelet(signal, 2**2)
ax[0].plot(signal)
ax[1].plot(z, 'r')
plt.show()

signal = df[df["classes"] == 'A'].iloc[0]["signals"][0:500]
fig, ax = plt.subplots(2,1)
fig.suptitle("AF Rhythm (A)")
z = apply_wavelet(signal, 2**2)
ax[0].plot(signal)
ax[1].plot(z, 'r')
plt.show()

signal = df[df["classes"] == 'O'].iloc[0]["signals"][0:500]
fig, ax = plt.subplots(2,1)
fig.suptitle("Other Rhythm (O)")
z = apply_wavelet(signal, 2**2)
ax[0].plot(signal)
ax[1].plot(z, 'r')
plt.show()


# %%
x = []
for i in signals:
  val = apply_wavelet(i, 2**2)
  x.append(val)

# %% [markdown]
# ### 2.4 Janelamento dos dados de treino

# %%
intersection = 100
window_size = 500
windows_array = ["max","max","max","max"]
classes_array = ['N', 'O', 'A', '~']

# %%
def window(signal, intersection, window_size, n):
  s_size = len(signal)
  max_n = (s_size-intersection)/(window_size - intersection)
  samples = []
  
  if  n == "max" or n > max_n:
    n = max_n
    n = math.floor(n)
    
  max_begin = s_size - (n*(window_size) - (n-1)*intersection)
  init = random.randrange(max_begin)
  for i in range(0, n):
    begin = init + i*window_size - i*intersection
    end = begin + window_size
    samples.append(signal[begin:end])

  
  return samples

# %%
def embalance_data(df, classes_array, windows_array, intersection, window_size):

  a = df.groupby('classes')
  d_train = df.copy()
  d_train = d_train.iloc[0:0]


  idx = -1
  for i in classes_array:
    aux = []
    idx += 1

    for index, row in a.get_group(i).iterrows():
      samples = window(row["signals"], intersection, window_size,  windows_array[idx])
      
      for k in range(len(samples)):
        d_train = d_train.append({'signals': samples[k], 'classes': i}, ignore_index=True)

  return d_train

# %%
x_train, x_test, y_train, y_test = train_test_split(x, df["classes"], test_size=0.25)
d_aux = pd.DataFrame({'signals': x_train, 'classes': y_train})

d_train = embalance_data(d_aux, classes_array, windows_array, intersection, window_size)

b = d_train.groupby('classes')

for i in ['N','O', 'A', '~']:
  print(i)
  aux = []

  for j in b.get_group(i)["signals"]:
    aux.append(len(j))

  print('Média:', round(np.mean(aux), 2))
  print("Std:", round(np.std(aux), 2))
  print("Qntd:", len(aux))
  print("Min:", min(aux))
  print("---------------")

# %%
plt.rcParams.update({'font.size': 20})

d_train["classes"].hist(
    figsize=(8, 6),
    grid = False,
    bins = 10,
)
plt.xticks(range(4))
plt.show()

# %%
d_train = d_train.sample(frac=1, random_state=1)
signals_2 = d_train["signals"]
classes_2 = d_train["classes"]

# %% [markdown]
# ### 2.4 "Normalização" do tamanho dos sinais

# %%
# for i in range(0, len(x)):
#   val = normalize([x[i]])
#   x[i] = val[0]
# 3. Implementação da CNN

# %% [markdown]
# # 3. Implementação do modelo

# %%
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl

# %%
y_train = classes_2
x_train = list(signals_2)

x_train = np.array(x_train)
# x_valid = np.array(x_valid)
x_test = np.array(x_test)

# %%
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# # x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
# x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)

# %%
# Import packages
from sklearn.model_selection import cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from math import floor
from sklearn.metrics import make_scorer, accuracy_score

pd.set_option("display.max_columns", None)

from sklearn import metrics
import tensorflow as tf

# %%
score_acc = make_scorer(accuracy_score)

# %%
def create_cnn(x_train, y_train):
  
  #x_train = np.array(list(x_train))
  x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
  y_train = pd.get_dummies(y_train)
  print(y_train.columns)

  input_shape = len(x_train[0])
  k1 = 5
  k2 = 3
  n = math.floor(input_shape/(k1+k2-2))
  nn_input = input_shape % (k1+k2-2)

  classificador = Sequential()

  for i  in range(0, 5):
    if i == 0:
      classificador.add(Conv1D(filters=64, kernel_size = k1, input_shape = (input_shape, 1), activation = "relu"))
    else:
        classificador.add(Conv1D(filters=64, kernel_size = k1, activation = "relu"))
    
    classificador.add(Conv1D(filters=32, kernel_size = k2, activation = "relu"))
    classificador.add(MaxPooling1D(pool_size=2))

  classificador.add(Flatten())

  classificador.add(Dense(units=64, activation='relu'))
  classificador.add(Dense(units=32, activation='relu'))
  classificador.add(Dense(units=64, activation='relu'))

  classificador.add(Dense(2, activation="softmax"))

  classificador.compile(loss='categorical_crossentropy', optimizer=Adamax(learning_rate=0.01), metrics=["accuracy"])


  es = EarlyStopping(monitor="accuracy", mode='max', verbose=0, patience=15)
  classificador.fit(x_train, y_train, epochs=150, batch_size=128, callbacks=es, verbose=1)
  
  return y_train.columns, classificador

# %% [markdown]
# # 4. Implementação da árvore

# %%
# x_train = x_train[0:200]
# y_train = y_train[0:200]

# %%
class RootNode:
  def __init__(self):
    self.child = None

# %%
class BinaryTreeNode:
  def __init__(self, model, classe):
    self.columns = None
    self.classe = classe
    self.model = model
    self.leftChild = None
    self.rightChild=None

# %%
def build_tree(x_train, y_train, classes_array, father):  

  if len(classes_array) < 2:
    print("impossivel classificar")
    return

  size = []

  for i in classes_array:
    count = 0
    for j in y_train:
      if j == i:
        count += 1
        
    size.append(count)

  index = size.index(max(size))
  majority = classes_array[index]
  del classes_array[index]

  if len(classes_array) != 1:
    y_next = y_train.replace(classes_array, "nan")
  else:
    y_next = y_train

  columns, model = create_cnn(x_train, y_next)
  node = BinaryTreeNode(model=model, classe="nan")
  node.leftChild = BinaryTreeNode(model=None, classe=majority)
  node.columns = columns

  if type(father) == RootNode:
    father.child = node
  else:
    father.rightChild = node

  if len(classes_array) == 1:
    node.rightChild = BinaryTreeNode(model=None, classe=classes_array[0])
  else:
    aux_df = pd.DataFrame({"signals": list(x_train), "classes": y_train})
    aux_df.drop(aux_df[aux_df['classes'] == majority].index, inplace=True)
    x_next = np.array(list(aux_df["signals"]))
    y_next = aux_df["classes"]
    
    build_tree(x_next, y_next, classes_array, node) 

# %%
classes_array = []
[classes_array.append(item) for item in y_train if item not in classes_array]
root = RootNode()
build_tree(x_train, y_train, classes_array, root)

# %%
def make_prediction(current_node, samples):
    if type(current_node) == RootNode:
        current_node = current_node.child

    samples = window(signal, intersection, window_size, "max")
    samples = np.array(samples)
    samples = samples.reshape(samples.shape[0],samples.shape[1], 1)

    while current_node.model != None:
        model = current_node.model
        prediction = model.predict(samples)
        prediction = pd.DataFrame(prediction)
        prediction.columns = current_node.columns
        prediction = prediction.idxmax(axis=1)
        prediction = prediction.mode()[0]
        # print(prediction)

        if prediction == current_node.leftChild.classe:
            current_node = current_node.leftChild
        elif prediction == current_node.rightChild.classe:
            current_node = current_node.rightChild

    return current_node.classe

# %%
y_pred = []
for signal in x_test:
    prediction = make_prediction(root, signal)
    y_pred.append(prediction)

# %%
classes_list = []
for i in y_test:
    if i not in classes_list:
        classes_list.append(i)

# %%
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cf_matrix = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = classes_list)
cm_display.plot()
plt.savefig("test.png")

# %%
acc = accuracy_score(y_test, y_pred)

print("matriz de confusão:", cf_matrix)
print("labels:", classes_list)
print("acurácia:", acc)


