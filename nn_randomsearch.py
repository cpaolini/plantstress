#load_ext tensorboard
from time import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from tabulate import tabulate
import timeit
import os


os.environ["CUDA_VISIBLE_DEVICES"]="3" # second gpu

import tensorflow as tf
#from tensorflow.compat.v1 import ConfigProto
#conf = tf.ConfigProto()
#conf.gpu_options.per_process_gpu_memory_fraction=0.2
#session = tf.Session(config=conf)

# Allow memory growth for the GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

#tf.config.experimental.set_virtual_device_configuration(gpus[0], 
#   [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#   
#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#tf.config.LogicalDeviceConfiguration(memory_limit=1024)
#tf.config.gpu.set_per_process_memory_growth(True)
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Input,Dropout

from scikeras.wrappers import KerasRegressor
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
#from tensorflow.compat.v1 import ConfigProto
#from tensorflow.compat.v1 import InteractiveSession
#config = ConfigProto()
#config.gpu_options.allow_growth = True
#session = InteractiveSession(config=config)
#from keras.backend.tensorflow_backend import set_session
#
#gpu_options = tf.GPUOptions(visible_device_list="3")
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
#mirrored_strategy = tf.distribute.MirroredStrategy()
#session_config = tf.ConfigProto()
#session_config.gpu_options.allow_growth=True
#sess = tf.Session(config=session_config)

#rm -f ./logs/
start = timeit.default_timer()
df = pd.read_csv("Soil_Airtemp.csv")


dataset = df[['Volumetric Water Content','Electrical Conductivity (dS/m)','Depth (cm)','T (degrees celcius)','AirT']]

print("Dataset Description",dataset.describe())

x= dataset[['Volumetric Water Content','Electrical Conductivity (dS/m)','Depth (cm)','T (degrees celcius)']]
y= dataset['AirT']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25x0.8=0.2

print("Dataset size    :",dataset.shape[0])
print("Training   size :",X_train.shape[0])
print("Validation size :",X_val.shape[0])
print("Testing    size :",X_test.shape[0])
print("Shape of X_train" ,X_train.shape)
print("Shape of y_train" ,y_train.shape)

print("Tensorflow Version",tf.__version__)

#def ANN(optimizer = 'adam',neurons=8,batch_size=50,epochs=50,activation='relu',loss='mae',dropout=0.05,patience=50):
#    model = Sequential()
#    model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation=activation))
#    model.add(Dropout(rate=dropout))
#    model.add(Dense(neurons, activation=activation))
#    model.add(Dropout(rate=dropout))
#    model.add(Dense(neurons, activation=activation))
#    model.add(Dropout(rate=dropout))
##    model.add(Dense(neurons, activation=activation))
##    model.add(Dropout(rate=dropout))
##    model.add(Dense(neurons, activation=activation))
##    model.add(Dropout(rate=dropout))
#    model.add(Dense(1))
#    model.compile(optimizer = optimizer, loss=loss)
#    early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience
#    history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,callbacks = [early_stopping],verbose=0) #verbose set to 1 will show the training process
#    return model

#rf_params = {
#    'optimizer': ['adam','sgd'],
#    'activation': ['relu','tanh'],
#    'loss': ['mae'],
#    'batch_size': [128,256,512],
#    #'neurons':[8,16,32,64],
#    'neurons':[8,16,32,64,128,256,512,1024],
##    'epochs':[100],
##    'patience':[10],
##    'dropout':[0.1]
#    'epochs':[250,500,1000,2500,5000],
#    'patience':[25,50,100,250,500],
#    'dropout':[0.1,0.2]
#}
def ANN(optimizer = 'adam',neurons=8,batch_size=50,epochs=50,activation='relu',loss='mae',dropout=0.05,patience=50):
    model = Sequential()
    model.add(Dense(neurons, input_shape=(X_train.shape[1],), activation=activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(rate=dropout))
    model.add(Dense(neurons, activation=activation))
    model.add(Dropout(rate=dropout))
#    model.add(Dense(neurons, activation=activation))
#    model.add(Dropout(rate=dropout))
    model.add(Dense(1))
    model.compile(optimizer = optimizer, loss=loss)
    early_stopping = EarlyStopping(monitor="loss", patience = patience)# early stop patience
    history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,callbacks = [early_stopping],verbose=1) #verbose set to 1 will show the training process
    return model
rf_params = {
    'optimizer': ['adam','sgd'],
    'activation': ['relu','tanh'],
    'loss': ['mae'],
    'batch_size': [128,256,512,1024,2048],
    'neurons':[8,16,32,64,128,256,512,1024],
  #  'epochs':[250,500,1000,2500,5000,7500,10000,15000,20000],
    'epochs':[1000,2500,5000,7500,10000,15000,20000],
  #  'patience':[25,50,100,250,750,1000,1500,2000],
    'patience':[100,250,750,1000,1500,2000],
    'dropout':[0.5,0.1,0.15,0.2]
}
#rf_params = {
#    'optimizer': ['adam'],
#    'activation': ['relu'],
#    'loss': ['mae'],
#    'batch_size': [128],
#    'neurons':[1024],
#   'epochs':[50],
#   'patience':[2],   
#    'dropout':[0.5]
#}
clf = KerasRegressor(model=ANN,optimizer = 'adam',neurons=8,batch_size=50,epochs=50,activation='relu',loss='mae',dropout=0.05,patience=50,verbose=0)
#grid = RandomizedSearchCV(clf,rf_params, n_iter=1,cv = [(slice(None), slice(None))],verbose=0,error_score='raise')
#grid = RandomizedSearchCV(clf,rf_params, n_iter=10,cv = 5,verbose=0,error_score='raise')
grid = RandomizedSearchCV(clf,rf_params, n_iter=1000,cv = 5,verbose=1,error_score='raise')

#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir="nn_thesis_logs/{}".format(time()),)
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
history = grid.fit(X_train, y_train,validation_data =( X_val,y_val),callback =[tensorboard])

print("Hyperparameters of Keras Regressor",rf_params)
print("The best hyperparamters are",grid.best_params_)

#grid.summary()

pred_y_train = grid.predict(X_train)
pred_y_val = grid.predict(X_val)
pred_y_test= grid.predict(X_test)

#Training set
r11 = mean_squared_error(y_train,pred_y_train)
r12 = sqrt(mean_squared_error(y_train,pred_y_train))
r13 = mean_absolute_error(y_train,pred_y_train)

# Validation set
r21 = mean_squared_error(y_val,pred_y_val)
r22 = sqrt(mean_squared_error(y_val,pred_y_val))
r23 = mean_absolute_error(y_val,pred_y_val)

# Testing set
r31 = mean_squared_error(y_test,pred_y_test)
r32 = sqrt(mean_squared_error(y_test,pred_y_test))
r33 = mean_absolute_error(y_test,pred_y_test)      
          
#print("Training Accuracy")
#print("Validation Accuracy")
#print("Testing Accuracy")


data=[
     ["Training Set",r11,r12,r13],
     ["Validation Set",r21,r22,r23],
     ["Testing Set  ",r31,r32,r33],


    ]
columns=["Dataset","MSE","RMSE","MAE"]
print(tabulate(data,headers=columns, tablefmt="fancy_grid"))

stop = timeit.default_timer()
print("Run Time(seconds) -", stop-start)

best_model = grid.best_estimator_
best_model.model_.save("nn_randomsearch_3layer.h5")


