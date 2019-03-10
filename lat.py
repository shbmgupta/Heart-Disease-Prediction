#from sklearn.cross_validation import cross_val_score
#from functools import reduce
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten
from keras.layers.core import Activation
from keras.layers import LeakyReLU
#from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings("ignore")
#from GSS.iiGSS import GSS
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

from keras.models import model_from_json
import pandas as pd
import math

# coding: utf-8

# In[15]:


data=[]
data = pd.read_csv("/home/aryan/Desktop/heart/framingham.csv",delimiter=",")
data = data.dropna()
scaler = MinMaxScaler()
data_nm = scaler.fit_transform(data)

len(data)


# In[17]:


import random
import random
random.shuffle(data_nm)
random.shuffle(data_nm)
random.shuffle(data_nm)
data_nm = pd.DataFrame(data_nm,index = data.index,columns = data.columns)
rows = data['TenYearCHD'].count()
train_size = int(math.ceil(rows*0.8))
train_size = int(train_size)



# In[18]:




# In[16]:male	age	education	currentSmoker	cigsPerDay	BPMeds	prevalentStroke	prevalentHyp	diabetes	totChol	sysBP	diaBP	BMI	heartRate	glucose	TenYearCHD


train_data = pd.DataFrame({'male':data_nm['male'].iloc[0:train_size],'age':data_nm['age'].iloc[0:train_size],'education':data_nm['education'].iloc[0:train_size],'currentSmoker':data_nm['currentSmoker'].iloc[0:train_size],'cigsPerDay':data_nm['cigsPerDay'].iloc[0:train_size],'BPMeds':data_nm['BPMeds'].iloc[0:train_size],'prevalentStroke':data_nm['prevalentStroke'].iloc[0:train_size], 'prevalentHyp':data_nm['prevalentHyp'].iloc[0:train_size], 'diabetes':data_nm['diabetes'].iloc[0:train_size],'totChol':data_nm['totChol'].iloc[0:train_size],'sysBP':data_nm['sysBP'].iloc[0:train_size],'diaBP':data_nm['diaBP'].iloc[0:train_size],'BMI':data_nm['BMI'].iloc[0:train_size],'heartRate':data_nm['heartRate'].iloc[0:train_size],'glucose':data_nm['glucose'].iloc[0:train_size]},index=data_nm.index[0:train_size])

train_target = pd.DataFrame({'TenYearCHD':data_nm['TenYearCHD'].iloc[0:train_size]},index = data_nm.index[0:train_size])

test_data = pd.DataFrame({'male':data_nm['male'].iloc[train_size:rows+1],'age':data_nm['age'].iloc[train_size:rows+1],'education':data_nm['education'].iloc[train_size:rows+1],'currentSmoker':data_nm['currentSmoker'].iloc[train_size:rows+1],'cigsPerDay':data_nm['cigsPerDay'].iloc[train_size:rows+1],'BPMeds':data_nm['BPMeds'].iloc[train_size:rows+1],'prevalentStroke':data_nm['prevalentStroke'].iloc[train_size:rows+1], 'prevalentHyp':data_nm['prevalentHyp'].iloc[train_size:rows+1], 'diabetes':data_nm['diabetes'].iloc[train_size:rows+1],'totChol':data_nm['totChol'].iloc[train_size:rows+1],'sysBP':data_nm['sysBP'].iloc[train_size:rows+1],'diaBP':data_nm['diaBP'].iloc[train_size:rows+1],'BMI':data_nm['BMI'].iloc[train_size:rows+1],'heartRate':data_nm['heartRate'].iloc[train_size:rows+1],'glucose':data_nm['glucose'].iloc[train_size:rows+1]},index=data_nm.index[train_size:rows+1])

test_target = pd.DataFrame({'TenYearCHD':data_nm['TenYearCHD'].iloc[train_size:rows+1]},index=data_nm.index[train_size:rows+1])

scaler_for_predictions = MinMaxScaler()
scaler_for_predictions.fit(data['TenYearCHD'].values.reshape(-1,1))


# In[19]:


import keras
from keras.layers import Dense
from keras.models import Sequential


# In[20]:
from keras import regularizers

model=Sequential()
model.add(Dense(128,input_dim=15,init="glorot_uniform",activation="relu",kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.5))
model.add(Dense(100,init="glorot_uniform",activation="relu",kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.3))
model.add(Dense(64,init="glorot_uniform",activation="relu",kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.01)))
model.add(Dropout(0.6))
model.add(Dense(1,init="uniform",activation="sigmoid"))


# In[21]:


model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.0001), metrics=['accuracy'])


# In[22]:


model.summary()


# In[23]:


train_data


# In[24]:


import numpy as np
train_data=np.array(train_data)
train_target=np.array(train_target)


# In[25]:


# train_data_new=train_data.reshape(-1,*train_data.shape)
# train_target_new=train_target.reshape(-1,*train_target.shape)


# In[26]:


print(train_data.shape)
print(train_target.shape)


# In[27]:


history=model.fit(x=train_data,y=train_target,epochs=50,verbose=2,validation_split=0.2,shuffle=True,)

# In[28]:


import matplotlib.pyplot as plt


# In[29]:


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
plt.show()


# In[30]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("error.png")
plt.show()


# In[31]:


model.evaluate(x=np.array(test_data),y=np.array(test_target),verbose=2)


# In[43]:

x = pd.DataFrame({'male':0,'age':61,'education':3,'currentSmoker':1,'cigsPerDay':31,'BPMeds':0,'prevalentStroke':0,'prevalentHyp':1,
'diabetes':0,'totChol':225,'sysBP':150,'diaBP':95,'BMI':28.58,'heartRate':65,'glucose':103},index=[0])

#x=[1,	39,	4,	0,	0,	0,	0,	0,	0,	195,	106,	70,	26.97,	80,	77]0
#0	61	3	1	30	0	0	1	0	225	150	95	28.58	65	103	1
#0	46	3	1	23	0	0	0	0	285	130	84	23.1	85	85	0
#0	43	2	0	0	0	0	1	0	228	180	110	30.3	77	99	0
#0	63	1	0	0	0	0	0	0	205	138	71	33.11	60	85	1



y=train_target[1]
print(x,y)


# In[44]:


a=np.array(x)
a.shape


# In[45]:


a=a.reshape(15,)


# In[49]:


pred=model.predict_classes(np.array([a]))


# In[50]

print("\n>>")
print(pred)
print("\n>>")
print(pred[0])


