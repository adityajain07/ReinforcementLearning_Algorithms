# %%
import gym

# %%
gym.envs.register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.74
)

# %% [markdown]
# #### Policy Evaluation

# %%
# initializing the state value function with all zeros
no_states  = 16
no_actions = 4
vpi_s      = {}

for i in range(no_states):
    vpi_s[i] = 0
    
print('Initialization of state value function: ', json.dumps(vpi_s, indent=3))

# %%
discount   = 1         # discount factor
threshold  = 0.0001      # threshold for terminating policy evaluation
iterations = 0          # no. of iterations it took to converge
converged  = False      # flag to exit while loop when converged
env_model  = env.env.P  # model of the environment i.e. transition probabilities

while not converged:
    iterations += 1
    max_diff = 0
    for state in vpi_s.keys():
        cur_value    = vpi_s[state]
        vpi_s[state] = 0             # will be updated next
        
        for action in range(no_actions):
            reward       = env_model[state][action][0][2]
            next_state   = env_model[state][action][0][1]
            trans_prob   = env_model[state][action][0][0]            
            vpi_s[state] += random_policy[state][action]*trans_prob*(reward + discount*vpi_s[next_state])
            
        max_diff = max(max_diff, abs(cur_value - vpi_s[state]))
        
    if max_diff<threshold: 
        converged = True   

# %%
print('It took', iterations, 'iterations to converge.')
print('There is no discounting, i.e., discount factor is 1')
print('The threshold for stopping policy evaluation: ', threshold)
print('Vpi(s) after one sweep of policy evaluation: ', json.dumps(vpi_s, indent=3))
print('The policy function is still random: ', json.dumps(random_policy, indent=3))

# %% [markdown]
# #### Policy Iteration

# %%
# intializing a arbitrary value function
import copy
import numpy as np
import json

no_states  = 16
no_actions = 4
Vpi_s      = {}

for i in range(no_states):
    Vpi_s[i] = 0

# %%
# intializing a random policy function
Policy = {}

for i in range(16):
    Policy[i] = [0.25, 0.25, 0.25, 0.25]

# %%
def policy_evaluation(discount, threshold, env_model):
    '''
    performs one sweep of policy evaluation
    '''
    global Vpi_s, Policy
    
    discount   = discount   # discount factor
    threshold  = threshold  # threshold for terminating policy evaluation
    iterations = 0          # no. of iterations it took to converge
    converged  = False      # flag to exit while loop when converged
    env_model  = env_model  # model of the environment i.e. transition probabilities
    no_actions = 4

    while not converged:
        iterations += 1
        max_diff = 0
        for state in Vpi_s.keys():
            cur_value    = copy.copy(Vpi_s[state])
            Vpi_s[state] = 0             # will be updated next
        
            for action in range(no_actions):
                reward       = env_model[state][action][0][2]
                next_state   = env_model[state][action][0][1]
                trans_prob   = env_model[state][action][0][0]            
                Vpi_s[state] += Policy[state][action]*trans_prob*(reward + discount*Vpi_s[next_state])
            
            max_diff = max(max_diff, abs(cur_value - Vpi_s[state]))
        
        if max_diff<threshold: 
            converged = True 
            
    return iterations
            
def policy_improvement(discount, env_model):
    '''
    performs one sweep of policy improvement
    '''
    global Vpi_s, Policy
    
    converged  = True
    no_actions = 4
    
    for state in Vpi_s.keys():
        cur_stateaction = copy.copy(Policy[state])   # current qpi(s,a)
        
        qpi_list = []                     # contains q(s,a) for every action
        for action in range(no_actions):
            reward       = env_model[state][action][0][2]
            next_state   = env_model[state][action][0][1]
            trans_prob   = env_model[state][action][0][0]
            qpi_list.append(trans_prob*(reward + discount*Vpi_s[next_state]))
            
        maxa_list      = np.argwhere(qpi_list == np.amax(qpi_list))
        maxa_list_indx = []
        
        # indices that have max. q values
        for item in maxa_list:
            maxa_list_indx.append(item[0])
            
        # updating the policy
        for i in range(no_actions):
            if i in maxa_list_indx:
                Policy[state][i] = 1/len(maxa_list_indx)
            else:
                Policy[state][i] = 0 
                
        if Policy[state]!=cur_stateaction:
            converged = False
            
    return converged
        

# %%
# Policy Iteration Loop
p_iterations = 0        # no of policy iteration steps
eval_iter    = 0        # total no of evaluation iterations
discount     = 0.9
threshold    = 0.0001

while True:
    eval_steps   = policy_evaluation(discount, threshold, env.env.P)
    converged    = policy_improvement(discount, env.env.P)
    p_iterations += 1    
    eval_iter    += eval_steps
    
    if converged:
        break 
    

# %%
print('Final Policy: ', json.dumps(Policy, indent=3))
print('Final State-Value Function: ', json.dumps(Vpi_s, indent=3))
print('Total number of policy iteration steps: ', p_iterations)
print('Total number of policy evaluation steps: ', eval_iter)

# %%



