#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pycaret
pycaret.__version__


# In[2]:


import numpy as np
import pandas as pd


# In[3]:


data = pd.read_csv('kr-vs-kp.data', delimiter=',')
data


# In[4]:


data.columns


# Setup

# In[5]:


# setup data
from pycaret.classification import *
s = setup(data, target = 'won', session_id = 123)


# In[6]:


# check available configs
get_config()


# In[7]:


# demonstrate X_train_transformed
get_config('X_train_transformed')


# In[8]:


# setup with minmax method of normalization
s = setup(data, target = 'won', session_id = 123, normalize = True, normalize_method = 'minmax')


# In[9]:


# determine effect X_train_transformed has on x-axis of fixed acidity param
get_config('X_train_transformed')['n.1'].hist()


# In[10]:


# compare x-axis to X_train
get_config('X_train')['n.1'].hist()


# Compare Models

# In[11]:


best = compare_models()


# In[12]:


# determine possible models
models()


# In[13]:


# sort models by best F1 score
best_F1_models_top3 = compare_models(sort = 'F1', n_select = 3)


# In[14]:


# list the best 3 models regarding F1 score
best_F1_models_top3


# Create Model

# In[15]:


# create model with best F1 score
cb = create_model('catboost')


# In[16]:


cb_results = pull()
print(type(cb_results))
cb_results


# In[17]:


# train catboost with specific number of folds
cb = create_model('catboost', fold = 5)


# In[18]:


# train model while returning train score as well as CV
create_model('catboost', return_train_score = True)


# Tune Model

# In[19]:


# default catboost model
cb = create_model('catboost')


# In[20]:


# tune hyperparameters of catboost model
tune_cb = tune_model(cb)


# In[21]:


# determine tuning grid
cb_grid = {'max_depth' : [None, 1, 2, 3, 4]}

# tune the xgboost model with tuning grid and F1 score
tuned_cb = tune_model(cb, custom_grid = cb_grid, optimize = 'F1')


# In[22]:


# access tuner object
tuned_cb, tuner = tune_model(cb, return_tuner = True)


# In[23]:


# model object
tuned_cb


# In[24]:


# tuner object
tuner


# In[25]:


# use optuna to tune model
tuned_cb = tune_model(cb, search_library = 'optuna')


# Ensemble Models

# In[26]:


# ensemble model through bagging
ensemble_model(cb, method = 'Bagging')


# In[27]:


# ensemble model through boosting
ensemble_model(cb, method = 'Boosting')


# Blend Models

# In[28]:


# blend the three models with best F1 score
blend_3 = blend_models(best_F1_models_top3)


# Stack Models

# In[29]:


# stack the best three models
stack_3 = stack_models(best_F1_models_top3)


# Plot and Interpret Model

# In[30]:


# plot the class report
plot_model(blend_3, plot = 'class_report', scale = 1)


# In[31]:


# save the class report
plot_model(blend_3, plot = 'class_report', scale = 2, save = True)


# In[32]:


# plot confusion matrix
plot_model(blend_3, plot='confusion_matrix')


# In[33]:


# plot AUC
plot_model(blend_3, plot = 'auc')


# In[34]:


# plot feature importance
plot_model(cb, plot = 'feature')


# In[35]:


# plot summary model
interpret_model(cb, plot = 'summary')


# Prediction

# In[36]:


# prediction on test set
holdout_pred = predict_model(blend_3)
holdout_pred.head()


# Get Leaderboard

# In[37]:


# Get leaderboard of models
lb = get_leaderboard()
lb


# In[38]:


# determine best model by F1 score from leaderboard
lb.sort_values(by = 'F1', ascending = False)['Model'].iloc[0]


# AutoML

# In[39]:


automl()


# Create App

# In[40]:


# Creat a gradio app
create_app(blend_3)


# Create API

# In[41]:


create_api(blend_3, api_name = 'EE8603_Final_Project')


# Create Docker

# In[42]:


create_docker('EE8603_Final_Project')


# In[ ]:


# %load DockerFile


FROM python:3.8-slim

WORKDIR /app

ADD . /app

RUN apt-get update && apt-get install -y libgomp1

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "EE8603_Final_Project.py"]


# In[ ]:


# %load requirements.txt

pycaret
fastapi
uvicorn


# Finalize Model

# In[45]:


final_best = finalize_model(blend_3)


# In[46]:


final_best


# Deploy Model

# In[47]:


# deploy model to cloud through aws s3
deploy_model(final_best, model_name = 'Final_Project_Truffen_Ryan',
             platform = 'aws', authentication = {'bucket' : 'truffen-ryan-final'})


# Save/Load Model

# In[50]:


# save the model
save_model(final_best, 'EE8603_Final_Project_model')


# In[51]:


# Load the model
loaded_from_disk = load_model('EE8603_Final_Project_model')
loaded_from_disk


# In[ ]:




