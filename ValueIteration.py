from Env import EnvLabyrinthe
import numpy as np
import time




def value_iteration_test_mdp(
    discount_factor : float,
    nb_lignes : int,
    nb_colonnes : int,
    alea : int
):

    start_time = time.time()

    env = EnvLabyrinthe(
        nblignes = nb_lignes,
        nbcolonnes = nb_colonnes,
        alea = alea # 0 : deterministe / 1 : stochastique
    )

    MDP = env.getMDP()

    NEW_STATE,PROBA,REWARD,DONE = 0,1,2,3

    # nb_actions = env.nb_actions 
    nb_observations = env.nb_states

    state_value_function = np.zeros(nb_observations)
    policy = np.zeros(nb_observations)


    theta = 1e-5

    delta = 1

    iteration = 0

    while delta > theta :
        # print(f"iteration {iteration} - delta : {delta}") 
        delta = 0
        for state in range(nb_observations) :
            v = state_value_function[state]
            # print(state)
            # print("test : ",MDP[state])
            data = MDP[state][0]

            probas = np.asarray([i[PROBA] for i in data])
            rewards = np.asarray([i[REWARD] for i in data])
            value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
            dones = np.asarray([i[DONE] for i in data])
            best_value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
            
            for action in range(len(MDP[state])) : 
                # print(len(MDP[state]), MDP[state][action])
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


    # print(state_value_function)


    # print("iteration valeur fait")

    for state in range(nb_observations) :
        old_action = policy[state]

        best_action = 0
        data = MDP[state][best_action]

        probas = np.asarray([i[PROBA] for i in data])
        rewards = np.asarray([i[REWARD] for i in data])
        value_new_states = np.asarray([state_value_function[i[NEW_STATE]] for i in data])
        dones = np.asarray([i[DONE] for i in data])
        best_value_action = (probas * (rewards + discount_factor * (1-dones) * value_new_states)).sum()
        for action in range(len(MDP[state])) : 
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

    end_time = time.time()


    return (end_time - start_time)


"""
time_training = value_iteration_test_mdp(
    discount_factor = 1.0,
    nb_lignes = 15,
    nb_colonnes = 20,
    alea = 0
)
print(f"time_training : {time_training}s")
"""
# print(policy)
#print("politique calculée")

# print(state_value_function)

""" 
# Test
state = env.replace_player_init()
env.render()
done = False
cum_sum = 0
while not done : 
    new_state,reward,done = env.step(int(policy[state]))
    cum_sum += reward
    state = new_state
    
    # print("state : ",state)
    # input()
    
print("-------------")
env.render()
print(f"test reward : {cum_sum}")
"""