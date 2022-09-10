#!/usr/bin/env python
# coding: utf-8

# ## Function for SVM Feature

# ### Pip Installs

# In[1]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
import os
import shutil
from itables import init_notebook_mode

init_notebook_mode(all_interactive=True)


# In[2]:


from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:


import pickle


# In[8]:


clf = pickle.load(open("save_svm_feature.pkl", "rb")) # loading the model 


# In[9]:


def img_tagger(direc):
  def image_feature(direc):
    model = InceptionV3(weights='imagenet', include_top=False)
    features = [];
    img_name = [];
    if os.path.isdir(direc):
      for files in tqdm(os.listdir(direc)):
          fname=direc+'/'+files
          img=image.load_img(fname,target_size=(224,224))
          x = img_to_array(img)
          x=np.expand_dims(x,axis=0)
          x=preprocess_input(x)
          feat=model.predict(x)
          feat=feat.flatten()
          features.append(feat)
          img_name.append(files)
      return features,img_name
    else:
      files=direc.split("/")[-1]
      fname=direc
      img=image.load_img(fname,target_size=(224,224))
      x = img_to_array(img)
      x=np.expand_dims(x,axis=0)
      x=preprocess_input(x)
      feat=model.predict(x)
      feat=feat.flatten()
      features.append(feat)
      img_name.append(files)
      return features,img_name
  test_x=[]
  img_name=[]
  if os.path.isdir(direc):
    for files in tqdm(os.listdir(direc)):
      img_name.append(files)
      fname=direc+'/'+files
      img_features,im_name=image_feature(str(fname))
      test_x.append(img_features)
          
  else:
    files=direc.split("/")[-1]
    fname=direc
    img_name.append(files)
    img_features,im_name=image_feature(str(fname))
    test_x.append(img_features)
  test_x_1=np.array(test_x)
  nsamples, nx, ny = test_x_1.shape
  test_dataset = test_x_1.reshape((nsamples,nx*ny))
  test_y=clf.predict(test_dataset)
  Test_df = pd.DataFrame(img_name,columns=['image'])
  Test_df["label"]=test_y
  return Test_df


# In[13]:


result=img_tagger(r"C:\Users\VV315MY\OneDrive - EY\Desktop\OCR\Test_3_tables")


# In[14]:


result


# In[ ]:




