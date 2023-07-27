import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor,plot_tree
from sklearn.model_selection import RandomizedSearchCV
from tabulate import tabulate
import timeit
import pickle 
import matplotlib.pyplot as plt

start = timeit.default_timer()
df = pd.read_csv("Soil_Airtemp.csv")


dataset = df[['Volumetric Water Content','Electrical Conductivity (dS/m)','T (degrees celcius)','Depth (cm)','AirT']]

print("Dataset Description",dataset.describe())

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

 #   '''Correlation Matrix'''
corr = dataset.corr(method='spearman')
mask = np.triu(np.ones_like(corr, dtype=bool))
cormat = sns.heatmap(corr,mask= mask,annot=True, cmap='YlGnBu', linewidths=1, fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
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
#regressor = xgb.XGBRegressor(tree_method='gpu_hist',gpu_id=1)
regressor = xgb.XGBRegressor(tree_method='gpu_hist',gpu_id=1,eval_metric ='mae')

#ROUND 1
param_grid ={
              "max_depth"     :[2,4,8,16,32,64,128,256],
              "n_estimators"  :[500,1000,1500,2000,2500,3000,3500,4000,4500,5000],
              "learning_rate":[0.1,0.01,0.001,0.0001]
               }
#ROUND 2               
param_grid ={
              "max_depth"     :[2,4,8,16,32,64,128,256],
              "n_estimators"  :[500,1000,1500,2000,2500,3000,3500,4000,4500,5000],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }

#ROUND 3               
param_grid ={
              "max_depth"     :[8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[500,1000,1500,2000,2500,3000,3500,4000,4500,5000],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }
#ROUND 4              
param_grid ={
              "max_depth"     :[8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[100,200,300,400,500,600,700,800,900,1000],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }
               #ROUND 5
param_grid ={
              "max_depth"     :[8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }     
               #ROUND 6          
param_grid ={
              "max_depth"     :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[25,50,75,100,125,150,175,200,225,250,275,300],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }
               #ROUND 7
param_grid ={
              "max_depth"     :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[100,110,120,130,140,150,160,170,180,190,200],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
               }    
#ROUND 8
param_grid ={
              "max_depth"     :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[100,110,120,130,140,150,160,170,180,190,200],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              'min_child_weight':[1,2,3,4,5,6,7,8,9,10],
              'gamma'           :[0.5,1,5,10,20,30,40,50]
               }     
#ROUND 9
param_grid ={
              "max_depth"     :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              'min_child_weight':[1,2,3,4,5,6,7,8,9,10],
              'gamma'           :[0.1,0.2,0.3,0.4,.5,0.6,0.7,0.8,0.9,1,5,10]
               }
               #ROUND 10
param_grid ={
              "max_depth"     :[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
              "n_estimators"  :[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200],
              "learning_rate":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],
              'min_child_weight':[1,2,3,4,5,6,7,8,9,10],
              'gamma'           :[1,2,3,4,5,],
              'max_delta_step':[1,2,3,4,5,6,7,8,9,10],
              'alpha' :[1,2,3,4,5,6,7,8,9,10],
              'colsample_bytree':[0.4,0.8,1,1.5,2,2.5,3,3.5,4,4.5,5],
              
               }                        
#search = RandomizedSearchCV(regressor,param_grid,n_iter=80,cv = [(slice(None), slice(None))],verbose=0)
search = RandomizedSearchCV(regressor,param_grid,n_iter=1000,cv = 5,verbose=0)
#search = RandomizedSearchCV(regressor,param_grid,n_iter=10,cv = 3,verbose=0)
#search = RandomizedSearchCV(regressor,param_grid,n_iter=5,cv =3,verbose=0,error_score='raise')
search.fit(X_train,y_train,eval_set = [(X_train,y_train),( X_val,y_val)],verbose=0)
print("Hyperparameters of XGB Regressor",param_grid)
print("The best hyperparameters are",search.best_params_)
print("*******************************")
pred_y_test= search.predict(X_test)
# Testing set
r31 = mean_squared_error(y_test,pred_y_test)
r32 = sqrt(mean_squared_error(y_test,pred_y_test))
r33 = mean_absolute_error(y_test,pred_y_test)  
data=[
    # ["Decision Tree ( EC in dS/m)",r11,r12,r13],
    # ["Validation Set",r21,r22,r23],
     ["Testing Set  ",r31,r32,r33],
    ]
    
columns=["Dataset","MSE","RMSE","MAE"]
print(tabulate(data,headers=columns, tablefmt="fancy_grid"))

stop = timeit.default_timer()
print("Run Time(seconds) -", stop-start)

#pred_y_val = search.predict(X_val)


#Training set
#r11 = mean_squared_error(y_test,test_val_dt)
#r12 = sqrt(mean_squared_error(y_test,test_val_dt))
#r13 = mean_absolute_error(y_test,test_val_dt)

# Validation set
#r21 = mean_squared_error(y_val,pred_y_val)
#r22 = sqrt(mean_squared_error(y_val,pred_y_val))
#r23 = mean_absolute_error(y_val,pred_y_val)

    
          
#print("Training Accuracy")
#print("Validation Accuracy")
#print("Testing Accuracy")


filename ="best_xgboost_model.pickle"
pickle.dump(search.best_estimator_,open(filename,'wb'))

loadmodel=pickle.load(open(filename,'rb'))
predictedvalue=loadmodel.predict(X_test)
print("MAE:",mean_absolute_error(y_test,predictedvalue))

fig, ax = plt.subplots(figsize=(30, 30))
plot_tree(search.best_estimator_,num_trees=5, ax=ax)
plt.show()