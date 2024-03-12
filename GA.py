import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from Rl import RLAlgo
np.random.seed(42)


class GA():
    def __init__(self, fitness, D, P, job, machine, operation, table_np, table_pd,
                 G=100, GS=0.6, LS=0.3, RS=0.1, pc=0.7, pTPX=0.5, pUX=0.5, pm=0.01):
        self.fitness = fitness
        self.D = D
        self.P = P
        self.G = G
        self.job = job
        self.machine = machine
        self.operation = operation
        self.GS = GS
        self.LS = LS
        self.RS = RS
        self.pc = pc
        self.pTPX = pTPX
        self.pUX = pUX
        self.pm = pm
        self.table_np = table_np
        self.table_pd = table_pd

        self.X_pbest = np.zeros([self.P, self.D])
        self.F_pbest = np.zeros([self.P]) + np.inf
        self.X_gbest = np.zeros([self.D])
        self.F_gbest = np.inf
        self.loss_curve = np.zeros(self.G)

        # Initialize RL parameters
        self.num_states=20
        self.num_actions=10
        self.e_greedy = 0.85

    def opt(self):
        # creating initial population
        self.X = self.initial_population()

        # calculating fitness value
        self.F = self.fitness(self.X, self.table_np, self.table_pd)

        # Defining cuurent state and action 
        self.rlObj=RLAlgo(self.num_states,self.num_actions,self.e_greedy)
        self.initial_fitness_scores=self.F
        curr_state = random.choice(range(self.num_states))
        curr_action_Pc=random.choice(range(self.num_actions))
        curr_action_Pm=random.choice(range(self.num_actions))

        t=0
        
        self.prev_fit_max=max(self.F)
        self.prev_fitness_scores=self.F
        
        # slga loop
        for g in range(self.G):
            st = time.time()
            reward_rc=self.rlObj.calculate_reward_rc(self.F,self.prev_fit_max)
            reward_rm=self.rlObj.calculate_reward_rm(self.F,self.prev_fitness_scores)

            # calculate next state
            next_state=self.rlObj.calculate_next_state(self.F,self.initial_fitness_scores,self.P)

            # conversion criteria
            if g<((self.num_states*self.num_actions)/2):
            # if True:    
                # SARSA ALGO
                next_action_Pc=self.rlObj.choose_action_pc(curr_state)
                next_action_Pm=self.rlObj.choose_action_pm(curr_state)

                curr_state, curr_action_Pc,curr_action_Pm,next_state,next_action_Pc,next_action_Pm = int(curr_state),int(curr_action_Pc),int(curr_action_Pm),int(next_state),int(next_action_Pc),int(next_action_Pm)

                self.rlObj.updateQTable_PC_SARSA(curr_state, curr_action_Pc,reward_rc,next_state,next_action_Pc)
                self.rlObj.updateQTable_PM_SARSA(curr_state, curr_action_Pm,reward_rm,next_state,next_action_Pm)

            else:
                # Q Learning ALGO
                next_action_Pc=self.rlObj.choose_action_pc(curr_state,True)
                next_action_Pm=self.rlObj.choose_action_pm(curr_state,True)

                curr_state, curr_action_Pc,curr_action_Pm,next_state,next_action_Pc,next_action_Pm = int(curr_state),int(curr_action_Pc),int(curr_action_Pm),int(next_state),int(next_action_Pc),int(next_action_Pm)
                
                self.rlObj.updateQTable_PC_QLearn(curr_state, curr_action_Pc,reward_rc,next_state,next_action_Pc)
                self.rlObj.updateQTable_PM_QLearn(curr_state, curr_action_Pm,reward_rc,next_state,next_action_Pm)

            
            self.prev_fit_max=max(self.F)
            self.prev_fitness_scores=self.F
            curr_state=next_state
            curr_action_Pc=next_action_Pc
            curr_action_Pm=next_action_Pm

            # select new pc and pm
            action_range_Pc=self.rlObj.get_action_set_Pc()[curr_action_Pc]
            start=action_range_Pc[0]
            end=action_range_Pc[1]

            self.pc=random.uniform(start,end)

            action_range_Pm=self.rlObj.get_action_set_Pm()[curr_action_Pm]
            start=action_range_Pm[0]
            end=action_range_Pm[1]

            self.pm=random.uniform(start,end)

            # genetic operations

            if np.min(self.F) < self.F_gbest:
                idx = self.F.argmin()
                self.X_gbest = self.X[idx].copy()
                self.F_gbest = self.F.min()
            self.loss_curve[g] = self.F_gbest

            # selection operation
            self.X = self.selection_operator()

            # crsoover operation
            self.crossover_operator()

            # mutation operation
            self.mutation_operator()

            # claculating best values
            ssst = time.time()
            self.F = self.fitness(self.X, self.table_np, self.table_pd)
            print("crossover rate",self.pc)
            print("mutation rate",self.pm)
            print("fitness",self.F)
            eeed = time.time()
            ed = time.time()
            cost = ed - st
            print(f'Iteration : {g + 1}')
            print(f'F_gbest : {self.F_gbest}')
            print(f'cost : {cost}')
            print(f'{eeed - ssst}')
            print('-' * 20 + '\n')

        self.rlObj.print_QtablePC()
        self.rlObj.print_QtablePM()


    def initial_population(self):
        X1 = []

        # initialize : MS
        P_gs = int(self.P * self.GS)
        for idx_chromosome in range(P_gs):
            chromosome = self.global_selection()
            X1.append(chromosome)

        P_ls = int(self.P * self.LS)
        for idx_chromosome in range(P_ls):
            chromosome = self.local_selection()
            X1.append(chromosome)

        P_rs = int(self.P * self.RS)
        for idx_chromosome in range(P_rs):
            chromosome = self.random_selection()
            X1.append(chromosome)

        # initialize : OS
        spam = self.table_pd['job'].copy().values
        X2 = []
        for i in range(self.P):
            np.random.shuffle(spam)
            X2.append(spam.copy())

        # merge : MS、OS
        X1 = np.array(X1)
        X2 = np.array(X2)
        X = np.hstack([X1, X2])

        return X

    # Initialize 1 (global selection)
    def global_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = pd.DataFrame(columns=['Machine_Selection', 'job', 'operation'])
        time_array = np.zeros(self.machine)

        for idx_job in sequence:
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask].reset_index(drop=True)

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-3].values
                added_time = time_array + processing_time
                selected_machine = added_time.argmin()
                time_array[selected_machine] = added_time[selected_machine]
                data = {'Machine_Selection': selected_machine,
                        'job': idx_job,
                        'operation': idx}
                # MS = pd.concat([MS, data], ignore_index=True)
                # Convert the dictionary to a DataFrame
                data_df = pd.DataFrame([data])  # Wrap the dictionary in a list to create a single-row DataFrame

                # Concatenate MS and data_df
                MS = pd.concat([MS, data_df], ignore_index=True)
                

        MS.sort_values(by=['job', 'operation'], inplace=True)
        MS.reset_index(drop=True, inplace=True)
        MS = MS['Machine_Selection'].tolist()

        return MS

    # Initialize 2 (local selection)
    def local_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = pd.DataFrame(columns=['Machine_Selection', 'job', 'operation'])

        for idx_job in sequence:
            time_array = np.zeros(self.machine)
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask].reset_index(drop=True)

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-3].values
                added_time = time_array + processing_time
                selected_machine = added_time.argmin()
                time_array[selected_machine] = added_time[selected_machine]
                data = {'Machine_Selection': selected_machine,
                        'job': idx_job,
                        'operation': idx}
                # MS = pd.concat([MS, data], ignore_index=True)
                # Convert the dictionary to a DataFrame
                data_df = pd.DataFrame([data])  # Wrap the dictionary in a list to create a single-row DataFrame

                # Concatenate MS and data_df
                MS = pd.concat([MS, data_df], ignore_index=True)

        MS.sort_values(by=['job', 'operation'], inplace=True)
        MS.reset_index(drop=True, inplace=True)
        MS = MS['Machine_Selection'].tolist()

        return MS

    # initialize 3 (random selection)
    def random_selection(self):
        sequence = np.random.choice(self.job, size=self.job, replace=False)
        MS = pd.DataFrame(columns=['Machine_Selection', 'job', 'operation'])

        for idx_job in sequence:
            mask = self.table_pd['job'] == idx_job
            table = self.table_pd[mask].reset_index(drop=True)

            for idx, row in table.iterrows():
                processing_time = row.iloc[:-3].values
                spam = np.where(processing_time != np.inf)[0]
                selected_machine = np.random.choice(spam, size=1)[0]
                data = {'Machine_Selection': selected_machine,
                        'job': idx_job,
                        'operation': idx}
                # MS = MS.append(data, ignore_index=True)
                # Convert the dictionary to a DataFrame
                data_df = pd.DataFrame([data])  # Wrap the dictionary in a list to create a single-row DataFrame

                # Concatenate MS and data_df
                MS = pd.concat([MS, data_df], ignore_index=True)

        MS.sort_values(by=['job', 'operation'], inplace=True)
        MS.reset_index(drop=True, inplace=True)
        MS = MS['Machine_Selection'].tolist()

        return MS


    def selection_operator(self):
        X_new = np.zeros_like(self.X)
        for i in range(self.P):
            X_new[i] = self.tournament()

        return X_new

    #  (tournament selection)
    def tournament(self, num=3):
        mask = np.random.choice(self.P, size=num, replace=True)
        F_selected = self.F[mask]
        X_selected = self.X[mask]
        c1_idx = F_selected.argmin()
        c1 = X_selected[c1_idx]

        return c1


    def crossover_operator(self):
        for i in range(self.P):
            p = np.random.uniform()
            if p < self.pc:
                p_idx = np.random.choice(self.P, size=2, replace=False)
                p1 = self.X[p_idx[0]].copy()
                p2 = self.X[p_idx[1]].copy()

                # MS
                r = np.random.uniform()
                if r <= self.pTPX:
                    p1[:self.operation], p2[:self.operation] = self.TPX(p1[:self.operation], p2[:self.operation])
                else:
                    p1[:self.operation], p2[:self.operation] = self.UX(p1[:self.operation], p2[:self.operation])

                # OS
                p1[self.operation:], p2[self.operation:] = self.POX(p1[self.operation:], p2[self.operation:])

                self.X[p_idx[0]] = p1.copy()
                self.X[p_idx[1]] = p2.copy()

    #(two-point crossover)
    def TPX(self, p1, p2):
        D = len(p1)
        boundary = np.random.choice(D, size=2, replace=False)
        boundary.sort()
        start, end = boundary[0], boundary[1]

        # swap
        c1 = p1.copy()
        c2 = p2.copy()
        c1[start:end] = p2[start:end]
        c2[start:end] = p1[start:end]

        return c1, c2

    # (uniform crossover)
    def UX(self, p1, p2):
        # Randomly select positions to exchange
        D = len(p1)
        mask = np.random.choice(2, size=D, replace=True).astype(bool)

        # swap
        c1 = p1.copy()
        c2 = p2.copy()
        c1[mask] = p2[mask]
        c2[mask] = p1[mask]

        return c1, c2

    #  (precedence preserving order-based crossover, POX)
    def POX(self, p1, p2):
        # Extract unique values for all elements and shuffle
        operation_set = np.unique(np.hstack([p1, p2]))
        np.random.shuffle(operation_set)
        # initilaize
        D = len(p1)
        c1 = np.zeros(D) - 1
        c2 = np.zeros(D) - 1

        # Randomly select elements to retain
        spam1 = np.random.choice(2, size=len(operation_set), replace=True).astype(bool)
        Js1 = operation_set[spam1]
        mask1 = np.isin(p1, Js1)
        # swap
        c1[mask1] = p1[mask1]
        c1[~mask1] = p2[~np.isin(p2, Js1)]

        # Randomly select elements to retain
        spam2 = np.random.choice(2, size=len(operation_set), replace=True).astype(bool)
        Js2 = operation_set[spam2]
        mask2 = np.isin(p2, Js2)
        # swap
        c2[mask2] = p2[mask2]
        c2[~mask2] = p1[~np.isin(p1, Js2)]

        return c1, c2

    def mutation_operator(self):
        for i in range(self.P):
            p1 = self.X[i].copy()
            p1[:self.operation] = self.machine_mutation(p1[:self.operation])
            p1[self.operation:] = self.swap_mutation(p1[self.operation:])
            self.X[i] = p1.copy()

    # Mutation 1 
    def machine_mutation(self, p1):
        # Generate a random number for each position
        D = len(p1)
        c1 = p1.copy()
        r = np.random.uniform(size=D)

        for idx, val in enumerate(p1):
            # f the random number is less than or equal to the mutation rate, then mutate that position (place it on the machine with the minimum time).
            if r[idx] <= self.pm:
                alternative_machine_set = self.table_np[idx]
                shortest_machine = alternative_machine_set[:-2].argmin()
                c1[idx] = shortest_machine

        return c1

    # Mutation 2 (swap mutation)
    def swap_mutation(self, p1):
        # Generate a random number for each position.
        D = len(p1)
        c1 = p1.copy()
        r = np.random.uniform(size=D)

        for idx1, val in enumerate(p1):
            # If the random number is less than or equal to the mutation rate, then mutate the position (randomly swap with another position)
            if r[idx1] <= self.pm:
                idx2 = np.random.choice(np.delete(np.arange(D), idx1))
                c1[idx1], c1[idx2] = c1[idx2], c1[idx1]

        return c1

    def get_fitness_score(self):
        return self.F

    def plot_curve(self):
        plt.figure()
        plt.title('loss curve ['+str(round(self.gBest_curve[-1], 3))+']')
        plt.plot(self.gBest_curve, label='loss')
        plt.grid()
        plt.legend()
        plt.show()
