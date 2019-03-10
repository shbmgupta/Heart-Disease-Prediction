
# coding: utf-8

# In[15]:


f=open("/home/aryan/Desktop/heart/framingham.csv","r")
data=[]
for i in f:
	try:
		data.append(list(map(float,i.split(","))))
	except:
		count=1


# In[16]:

from sklearn import preprocessing as prp
len(data)
data_nm=data

# In[17]:


import random
random.shuffle(data)
random.shuffle(data)
random.shuffle(data)
train_split=int(len(data)*0.7)
train=data[:train_split]
test=data[train_split:]


# In[18]:


train_target=[]
train_data=[]
for i in train:
    train_target.append(int(i[-1]))
    train_data.append(i[:-1])
test_target=[]
test_data=[]
for i in test:
    test_target.append(int(i[-1]))
    test_data.append(i[:-1])


# In[19]:


import keras
from keras.layers import Dense
from keras.models import Sequential


# In[20]:


model=Sequential()
model.add(Dense(256,input_dim=15,init="uniform",activation="tanh"))
model.add(Dense(128,init="uniform",activation="tanh"))
model.add(Dense(1,init="uniform",activation="sigmoid"))


# In[21]:


model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.adam(lr=0.001), metrics=['accuracy'])


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


history=model.fit(x=train_data,y=train_target,epochs=150,verbose=2,validation_split=0.1,shuffle=True,)


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


x=[0,	38,	2,	1,	20,	0,	0,	1,	0,	221,	140,	90,	21.35,	95,	70
]
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

print("\n")
print(pred[0])


