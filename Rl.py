import numpy as np
import random
import pandas as pd

class RLAlgo:

    def __init__(self,num_states,num_actions,e_greedy):
        self.num_states=num_states
        self.num_actions=num_actions
        self.e_greedy=e_greedy
        self.alpha=0.75
        self.gamma=0.2

        self.state_set = [(0.0, 0.05), (0.05, 0.1), (0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35), (0.35, 0.4), (0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
        
        self.action_set_pc = [(0.4, 0.45), (0.45, 0.5), (0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7), (0.7, 0.75), (0.75, 0.8), (0.8, 0.85), (0.85, 0.9)]

        self.action_set_pm=[(0.01, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.09), (0.09, 0.11), (0.11, 0.13), (0.13, 0.15), (0.15, 0.17), (0.17, 0.19), (0.19, 0.21)]

        self.Q_table_pc = np.zeros((self.num_states, self.num_actions))
        self.Q_table_pm = np.zeros((self.num_states, self.num_actions))


    
    def get_action_set_Pc(self):
        return self.action_set_pc
    
    def get_action_set_Pm(self):
        return self.action_set_pm   

    def get_QTablePC(self):
        return self.Q_table_pc   
     
    def get_QTablePM(self):
        return self.Q_table_pm
           
    # Function to choose an action for pc using e-greedy policy
    def choose_action_pc(self, state, is_greedy=False):
       
        if is_greedy or np.random.rand() > self.e_greedy:
            # Exploitation: choose the action with the highest Q value
            return np.argmax(self.Q_table_pc[state, :])
        else:
            # Exploration: choose a random action
            return random.choice(range(len(self.action_set_pc)))

    # Function to choose an action for pm using e-greedy policy
    def choose_action_pm(self, state, is_greedy=False):
        if is_greedy or np.random.rand() > self.e_greedy:
            # Exploitation: choose the action with the highest Q value
            return np.argmax(self.Q_table_pm[state, :])
        else:
            # Exploration: choose a random action
            return random.choice(range(len(self.action_set_pm)))
        

    def calculate_reward_rc(self,fitness_scores,prev_max_fit):
        reward_rc=(max(fitness_scores)-prev_max_fit)/prev_max_fit
        return reward_rc


    def calculate_reward_rm(self,fitness_scores,prev_fitness_scores):
        reward_rm=(sum(fitness_scores)-sum(prev_fitness_scores))/sum(prev_fitness_scores)
        return reward_rm


    def calculate_next_state(self,fitness_scores,init_fitvalue,N):
        w1=0.35
        w2=0.35
        w3=0.3

        f=sum(fitness_scores)/sum(init_fitvalue)

        avg=sum(fitness_scores)/N
        mod_sum_N = sum(abs(x - avg) for x in fitness_scores)
        avg=sum(init_fitvalue)/N
        mod_sum_D = sum(abs(x - avg) for x in init_fitvalue)

        d=mod_sum_N/mod_sum_D

        m=max(fitness_scores)/max(init_fitvalue)

        next_state=w1*f+w2*d+w3*m

        if next_state>=0.9 :
            return 19

        for i, (lower, upper) in enumerate(self.state_set):
            if lower <= next_state < upper:
                next_state_idx=i
                break

        return next_state_idx

    # for pc
    # update Q table for sarsa
    def updateQTable_PC_SARSA(self,state,action,reward,nextState,nextAction):
        
        self.Q_table_pc[state, action] = (1-self.alpha)*self.Q_table_pc[state, action] + self.alpha * (reward + (self.gamma * self.Q_table_pc[nextState, nextAction]))

    # update Q table for q learn
    def updateQTable_PC_QLearn(self,state,action,reward,nextState,nextAction):
        print("qlearn")
        self.Q_table_pc[state, action] = (1-self.alpha)*self.Q_table_pc[state, action] + self.alpha * (reward + (self.gamma * max(self.Q_table_pc[nextState])))


    # for pm
    # update Q table for sarsa
    def updateQTable_PM_SARSA(self,state,action,reward,nextState,nextAction):
        self.Q_table_pm[state, action] = (1-self.alpha)*self.Q_table_pm[state, action] + self.alpha * (reward + (self.gamma * self.Q_table_pm[nextState, nextAction]))

    # update Q table for q learn
    def updateQTable_PM_QLearn(self,state,action,reward,nextState,nextAction):
        self.Q_table_pm[state, action] = (1-self.alpha)*self.Q_table_pm[state, action] + self.alpha * (reward + (self.gamma * max(self.Q_table_pm[nextState])))

    def print_QtablePC(self):
        qtbale=pd.DataFrame(self.Q_table_pc)

        print(qtbale)
    
    def print_QtablePM(self):
        qtbale=pd.DataFrame(self.Q_table_pm)

        print(qtbale)

        

            
