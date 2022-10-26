from Env import EnvLabyrinthe
import numpy as np

env = EnvLabyrinthe(
    nblignes = 5,
    nbcolonnes = 5,
    alea = 0 # 0 : deterministe / 1 : stochastique
)

MDP = env.getMDP()
env.init_grid()
NEW_STATE,PROBA,REWARD,DONE = 0,1,2,3



nb_actions = env.nb_actions 
nb_observations = env.nb_states

state_value_function = np.zeros(nb_observations)
policy = np.zeros(nb_observations)

discount_factor = 0.99

theta = 1e-10

delta = 1

iteration = 0


while delta > theta :
    print(f"iteration {iteration} - delta : {delta}") 
    delta = 0
    for state in range(nb_observations) :
        v = state_value_function[state]
        data = MDP[state][0]

        probas = np.asarray([i[PROBA] for i in data])
        rewards = np.asarray([i[REWARD] for i in data])
        value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
        dones = np.asarray([i[DONE] for i in data])
        best_value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
        
        for action in range(nb_actions) : 
            data = MDP[state][action]
            probas = np.asarray([i[PROBA] for i in data])
            rewards = np.asarray([i[REWARD] for i in data])
            value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
            dones = np.asarray([i[DONE] for i in data])
            value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
            if value_action > best_value_action :
                best_value_action = value_action
    
        state_value_function[state] = best_value_action
        delta = max(delta , abs(v - state_value_function[state]))
    iteration += 1

for state in range(nb_observations) :
    old_action = policy[state]

    best_action = 0
    data = MDP[state][best_action]

    probas = np.asarray([i[PROBA] for i in data])
    rewards = np.asarray([i[REWARD] for i in data])
    value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
    dones = np.asarray([i[DONE] for i in data])
    best_value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
    for action in range(nb_actions) : 
        data = MDP[state][action]
        probas = np.asarray([i[PROBA] for i in data])
        rewards = np.asarray([i[REWARD] for i in data])
        value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
        dones = np.asarray([i[DONE] for i in data])
        value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
        if value_action > best_value_action :
            best_value_action = value_action
            best_action = action

    policy[state] = best_action


print(state_value_function)



# Test
state = env.replace_player_init()

done = False
cum_sum = 0
while not done : 
    new_state,reward,done = env.step(int(policy[state]))
    cum_sum += reward
    state = new_state
    env.render()
    print("state : ",state)


print(f"test reward : {cum_sum}")
