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
# #1. Aquisição dos Dados

# %%
data = Path('data')

signals = []
classes =[]

for value in sorted(data.glob('*.mat')):
  signal = scipy.io.loadmat(value)
  signal = np.array(signal["val"], dtype='float')[0]
  signals.append(signal)
  # print(value)

# %%
classification_path = os.path.join(data, "classification.txt")
with open(classification_path) as f:
    classes = f.readlines()

for i in range(0, len(classes)):
  classes[i] = classes[i][classes[i].find(','):]
  classes[i] = classes[i][: classes[i].find('\n')]
  classes[i] = classes[i].replace(",", "")

#classes = classes[0: -1]

# %% [markdown]
# # 2. Preprocessamento

# %% [markdown]
# ### 2.1 Análise da distribuição das classes

# %%
d = {'signals': signals, 'classes': classes}
df = pd.DataFrame(data=d)
a = df.groupby('classes')

# %%
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
# plt.savefig('init_distri.png')
plt.show()

# %% [markdown]
# ### 2.3 Wavelet helping functions

# %%
def gera_filtro_MH(Escala):
  passo = 1/Escala;
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

# %%
def apply_wavelet(signal, level):
  Filtro = gera_filtro_MH(level)
  y = np.convolve(signal,Filtro)
  return y

# %% [markdown]
# ### 2.4 Aplicação da tranformada Wavelet

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
# ### 2.2 Janelamento dos dados de treino

# %%
intersection = 100
window_size = 500
windows_array = [1,2,6,20]
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
x_rem, x_test, y_rem, y_test = train_test_split(x, df["classes"], test_size=0.25)
x_train, x_valid, y_train, y_valid = train_test_split(x_rem, y_rem, test_size=0.1)

d_aux = pd.DataFrame({'signals': x_train, 'classes': y_train})
d_train = embalance_data(d_aux, classes_array, windows_array, intersection, window_size)

d_aux = pd.DataFrame({'signals': x_valid, 'classes': y_valid})
d_valid = embalance_data(d_aux, classes_array, windows_array, intersection, window_size)

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
# plt.savefig('end_distri.png')
plt.show()

# %% [markdown]
# ### 2.4 "Normalização" do tamanho dos sinais

# %%
# for i in range(0, len(x)):
#   val = normalize([x[i]])
#   x[i] = val[0]

# %% [markdown]
# # 3. Implementação da CNN

# %% [markdown]
# ## 3.1 Descoberta dos hyperparâmetros (Otimização Bayseana)

# %%
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv1D, MaxPooling1D

from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from bayes_opt import BayesianOptimization

# %%
y_train = pd.get_dummies(d_train["classes"])
y_valid = pd.get_dummies(d_valid["classes"])


x_train = np.array(list(d_train["signals"]))
x_valid = np.array(list(d_valid["signals"]))
x_test = np.array(list(x_test))

# %%
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
x_valid = x_valid.reshape(x_valid.shape[0], x_valid.shape[1], 1)

# %%
# Import packages
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
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
len(x_train[0])

# %%
def optmize_cnn(neurons, conv_layers, optimizer, filters):
  input_shape = len(x_train[0])
  k1 = 5
  k2 = 3
  neurons2 = math.floor(neurons/2)

  classificador = Sequential()

  for i in range(0, conv_layers):
    if i == 0:
      classificador.add(Conv1D(filters=filters, kernel_size = k1, input_shape = (input_shape, 1), activation = "relu"))
    else:
      classificador.add(Conv1D(filters=filters, kernel_size = k1, activation = "relu"))
    
    classificador.add(Conv1D(filters=filters, kernel_size = k2, activation = "relu"))
    classificador.add(MaxPooling1D(pool_size=2))

  classificador.add(Flatten())

  classificador.add(Dense(units=neurons, activation='relu'))
  classificador.add(Dense(units=neurons2, activation='relu'))
  classificador.add(Dense(units=neurons, activation='relu'))

  classificador.add(Dense(4, activation="softmax"))

  classificador.compile(loss='categorical_crossentropy', optimizer=optimizer(learning_rate=0.01), metrics=["accuracy"])

  return classificador

# %%
def cross_val_func(model, x_train, y_train, batch_size, folds):
  kf = KFold(n_splits=folds, random_state=1, shuffle=True)
  score = []
  
  for train_index, test_index in kf.split(x_train): 

    #print("TRAIN:", train_index, "TEST:", test_index)
    xtr, xte = x_train[train_index], x_train[test_index]
    ytr, yte = y_train.iloc[train_index], y_train.iloc[test_index]

    model.fit(xtr, ytr, epochs=150, batch_size=batch_size, verbose=0)

    y_pred = model.predict(xte)
    y_pred = (np.asarray(y_pred)).round()
    y_pred = y_pred.astype(int)
    yte = np.array(yte)

    acc = accuracy_score(yte, y_pred)
    
    score.append(acc)

  return score

# %%
def evaluate_network(neurons, conv_layers, optimizer, batch_size, filters):
    global patience
    global count
    global max_val

    if count == patience:
        return
    
    neurons = round(neurons)
    optimizer = round(optimizer)
    conv_layers = round(conv_layers)
    batch_size = 10*round(batch_size)
    filters = 2**round(filters)


    optimizer_array = [Adam, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl, SGD]
    optimizer_val = optimizer_array[optimizer]

    model = optmize_cnn(neurons, conv_layers, optimizer_val, filters)

    score = cross_val_func(model, x_train, y_train, batch_size, 5)

    score = sum(score)/len(score)

    if score > max_val:
        max_val = score
        count = 0
    else:
        count += 1

    return score

# %%
# pbounds = {'neurons': (8, 256),
#           'conv_layers': (1, 5),
#            'optimizer': (0, 7),
#            'batch_size': (6, 100),
#            'filters': (4, 7)
#         }

#optimizer = BayesianOptimization(
#    f=evaluate_network,
#    pbounds=pbounds,
#    verbose=2,  # verbose = 1 prints only when a maximum 
#    # is observed, verbose = 0 is silent
#    random_state=1
#)

patience = 20
max_val = 0
count = 0

import warnings
warnings.filterwarnings('ignore')
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

#optimizer.maximize(init_points=125, n_iter=75,)


# %%
#params_nn_ = optimizer.max['params']
#params_nn_['neurons'] = round(params_nn_['neurons'])
#params_nn_['batch_size'] = round(10*params_nn_['batch_size'])
#params_nn_['filters'] = 2**round(params_nn_['filters'])

optimizerL = [Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl, Adam]
#params_nn_['optimizer'] = optimizerL[round(params_nn_['optimizer'])]
#params_nn_['conv_layers'] = round(params_nn_['conv_layers'])
#params_nn_

# %%
#model = optmize_cnn(params_nn_['neurons'], params_nn_['conv_layers'], params_nn_['optimizer'],
#                    params_nn_['filters'])

#|   iter    |  target   | batch_... | conv_l... |  filters  |  neurons  | optimizer |
#| 19        | 0.8501    | 16.79     | 4.798     | 5.35      | 151.4     | 2.857     |

model = optmize_cnn(round(151.4), round(4.798), optimizerL[round(2.857)], 2**round(5.35))

model.fit(x_train, y_train, epochs=150, batch_size=10*round(16.79), verbose=0)

# %%
y_pred = model.predict(x_valid)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['A', 'N', 'O', '~']

# %%
y_valid = y_valid.idxmax(axis=1)
y_pred = y_pred.idxmax(axis=1)

# %%
accuracy_score(y_valid, y_pred)
# score = cross_val_score(classificador, x_valid, y_valid, scoring=score_acc, cv=5)

# media = score.mean()
# std = score.std()

# %%
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cf_matrix = confusion_matrix(y_valid, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = ['A', 'N', 'O', '~'])
cm_display.plot()
plt.savefig("valid.png")

# %% [markdown]
# ## 4. Verificando  a precisão com os dados de teste

# %%
y_pred = []
for signal in x_test:
  samples = window(signal, intersection, window_size, "max")
  samples = np.array(samples)
  samples = samples.reshape(samples.shape[0],samples.shape[1], 1)
  
  prediction = model.predict(samples)
  prediction = pd.DataFrame(prediction)
  prediction.columns = ['A', 'N', 'O', '~']
  prediction = prediction.idxmax(axis=1)
  val = prediction.mode()[0]
  y_pred.append(val)



# %%
# y_pred = classificador.predict(x_test)
# y_pred = pd.DataFrame(y_pred)
# y_pred.columns = ['A', 'N', 'O', '~']

# %%
# y_test = y_test.idxmax(axis=1)
# # y_pred = y_pred.idxmax(axis=1)

# %%
accuracy_score(y_test, y_pred)
# score = cross_val_score(classificador, x_valid, y_valid, scoring=score_acc, cv=5)

# media = score.mean()
# std = score.std()

# %%
from sklearn.metrics import confusion_matrix
from sklearn import metrics

cf_matrix = confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cf_matrix, display_labels = ['A', 'N', 'O', '~'])
cm_display.plot()
plt.savefig("test.png")

# %%
accuracy_score(y_test, y_pred)



# %%
