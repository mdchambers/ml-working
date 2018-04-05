
# coding: utf-8

# # Chapter 3 : Exercises
# 
# ## Exercise 1
# 
# *Try to build a classifier for the MNIST dataset that achieves over 97% accuracy on the test set. Hint: the KNeighborsClassifier works quite well for this task; you just need to find good hyperparameter values (try a grid search on the weights and n_neighbors hyperparameters).*

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shelve

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
mnist


# In[2]:


x, y = mnist["data"], mnist["target"]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]


# In[3]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.get_params()


# In[4]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'weights': ['uniform', 'distance'], 'n_neighbors': [3, 5, 7]}
]

print("Beginning gridsearch")
grid_search = GridSearchCV(knn_clf, param_grid, cv = 3, n_jobs = -1, verbose = 2)
grid_search


# In[ ]:


grid_search.fit(x_train, y_train)

print("Gridsearch done; writing state")
save_state = shelve.open("03_exercise_01_state_01")
save_state["grid"] = grid_search
save_state["knn_clf"] = knn_clf
save_state.close()

