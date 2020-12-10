#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Lets import the packages
import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_app_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

np.random.seed(1)


# ## Intialization

# ### 2-layer Neural Network

# In[2]:


# Initialize parameters
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1= np.random.randn(n_h,n_x) * 0.01
    b1= np.zeros((n_h,1))
    W2= np.random.randn(n_y,n_h) * 0.01
    b2= np.zeros((n_y,1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# In[3]:


parameters = initialize_parameters(3,2,1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# ### 3.2 - L-layer Neural Network
# 

# In[8]:


def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters={}
    for l in range(1,len(layer_dims)):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


# In[9]:


parameters = initialize_parameters_deep([5,4,3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))


# ## 4 - Forward propagation module

# ### 4.1 - Linear Forward 

# In[66]:


def linear_forward(A, W, b):
   
    Z = np.dot(W,A)+b
    cache = (A, W, b)
    return Z,cache


# In[67]:


A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))


# ### 4.2 - Linear-Activation Forward
# 

# In[109]:


def linear_activation_forward(A_prev, W, b, activation):
    if activation == 'sigmoid':
        A_current, activation_cache = sigmoid(linear_forward(A_prev,W,b)[0])
    elif activation == 'relu':
        A_current, activation_cache = relu(linear_forward(A_prev,W,b)[0])
    cache = (linear_forward(A_prev,W,b)[1], activation_cache)
    #cache= ((AWB),Z)
    return A_current, cache


# In[108]:


A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))


# In[70]:


linear_activation_cache


# ### d) L-Layer Model 

# In[75]:


def L_model_forward(X, parameters):
    A_prev = X
    L = len(parameters)//2
    caches = []
    for l in range(1,L):
        A_prev,linear_activation_cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation = "relu")
        caches.append(linear_activation_cache)


    
    Yhat, linear_activation_cache = linear_activation_forward(A_prev, parameters['W' + str(L)], parameters['b' + str(L)],activation='sigmoid')
    caches.append(linear_activation_cache)
    return (Yhat, caches)


# In[ ]:


#linear activation function: Acurr, Aprev,W, b, Zprev


# In[76]:


X.shape


# In[77]:


parameters


# In[78]:


X, parameters = L_model_forward_test_case_2hidden()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))


# ## 5 - Cost function

# In[82]:


def compute_cost(AL, Y):
    m = Y.shape[1]
    J = (-1/m)* np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))
    return J


# In[83]:


Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))


# ## 6 - Backward propagation module

# In[88]:


def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = (1/m)*(np.dot(dZ, A_prev.T))
    db = (1/m) * np.sum(dZ, axis = 1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[89]:


# Set up some test inputs
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# ### 6.2 - Linear-Activation backward

# In[122]:


# linear_activation_backward
def linear_activation_backward(dA, cache, activation):
    activation_cache = cache[1]
    print(activation_cache)
    if activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    elif activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, cache[0])
    return dA_prev, dW, db
    


# In[123]:


AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))


# ### 6.3 - L-Model Backward 
# 

# In[148]:


def L_model_backward(AL, Y, caches):
    grads={}
    L = len(caches)
    print(L)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dA_prev, dW, db = linear_activation_backward(dAL, caches[-1], activation='sigmoid')
    grads['dA'+str(L)] = dA_prev
    grads['dW'+str(L)] = dW
    grads['db'+str(L)] = db
    print(dA_prev)
    for l in range(L-1,0,-1):
        print(l)
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[l-1], activation='relu')
        grads['dA'+str(l)] = dA_prev
        grads['dW'+str(l)] = dW
        grads['db'+str(l)] = db
    
    return grads
        
    


# In[149]:


AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print_grads(grads)


# ### 6.4 - Update Parameters

# In[158]:


def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    print(L)
    for l in range(1,L+1):
        parameters['W'+str(l)] -= learning_rate * grads['dW'+str(l)]
        parameters['b'+str(l)] -= learning_rate * grads['db'+str(l)]
    return parameters


# In[159]:


parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))


# In[ ]:




