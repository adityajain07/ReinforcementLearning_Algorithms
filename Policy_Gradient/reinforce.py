
import matplotlib.pyplot as plt
import gym
from IPython import display as ipythondisplay
import numpy as np
import json
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch

# %%
env = gym.make('CartPole-v0')

print('Observation Space: ', env.observation_space)
print('Action Space: ', env.action_space)

no_actions = 2

# ##### Initialize the Neural Network
# 

# %%
input_dim  = 4              # cart position, cart velocity, pole angle, pole angular velocity
no_layers1 = 256
# no_layers2 = 128
no_layers3 = 128
out_dim    = no_actions     # no of actions

model = keras.Sequential()
model.add(keras.Input(shape=(input_dim,)))  
model.add(layers.Dense(no_layers1, activation="relu"))
# model.add(layers.Dense(no_layers2, activation="relu"))
model.add(layers.Dense(no_layers3, activation="relu"))
model.add(layers.Dense(out_dim, activation="softmax"))

model.summary()

# %% [markdown]
# ##### Action Selection Function

# %%
def select_action(obs, no_actions, model): 
    '''chooses action based on softmax over network's outputs'''
    
    obs     = tf.expand_dims(obs, axis=0)
    pred    = model(obs)
    pred    = pred.numpy()
    action  = np.random.choice(no_actions, p=pred[0])
        
    return action

# %% [markdown]
# ##### REINFORCE Implementation

# %% [markdown]
# The network is trained for 1000 episodes

# %%
env        = gym.make('CartPole-v0') 
converged  = False
episodes   = 0
alpha      = 1e-3
optimizer  = keras.optimizers.SGD(learning_rate=alpha)
gamma      = 0.9

while episodes<1000:
    episodes    += 1
    cur_obs     = env.reset()
    cur_action  = select_action(cur_obs, no_actions, model)
    done        = False
    
    trans_list  = []
    trans_list.append([cur_obs, cur_action])
    t_steps     = 0

    # interaction with env
    while not done:        
        next_obs, reward, done, info = env.step(cur_action)
        t_steps                      += 1
        next_action                  = select_action(next_obs, no_actions, model)        
        
        if done: 
            trans_list.append([reward])
        else:
            trans_list.append([next_obs, next_action, reward])        
            cur_obs    = next_obs  
            cur_action = next_action
            
    
    # updating model parameters
    steps = len(trans_list)
    G     = 0    
    for i in range(steps-2, -1, -1):
        G = trans_list[i+1][-1] + gamma*G
        
        with tf.GradientTape() as tape:
            prediction = model(tf.expand_dims(trans_list[i][0], axis=0), training=True) 
            loss       = -tf.math.log(prediction[0][trans_list[i][1]]) * G
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
    
    experiment.log_metric("Steps per episode", t_steps, step=episodes)
#     print('Episode no: ', episodes, ' - Steps survived: ', t_steps)

model.save('reinforce_e5.h5')
experiment.end()

# #### Check controller performance

# %%
def control_performance(env_name, model, no_actions, trials):    
    env          = gym.make(env_name)
    steps_list   = []
    
    for i in range(trials):
        done           = False
        cur_obs        = env.reset()
        cur_action     = select_action(cur_obs, no_actions, model)
        t_steps        = 0
        
        while not done:        
            next_obs, reward, done, info = env.step(cur_action)
            t_steps                      += 1
            next_action                  = select_action(next_obs, no_actions, model)  
            
            cur_obs    = next_obs  
            cur_action = next_action
            
        steps_list.append(t_steps)
        
    return steps_list

# %%
steps = control_performance('CartPole-v0', model, no_actions, 100)
print('Average steps the pole is sustained in 100 trials: ', sum(steps)/len(steps))





