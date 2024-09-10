import gym
import numpy as np
import random
from tensorflow import keras , math
from keras import layers , optimizers , Sequential 
import matplotlib.pyplot as plt


def draw_3D_chart(chart_data_time, chart_data_reward, chart_data_action, algorithm_name):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(chart_data_time, chart_data_reward, chart_data_action, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    ax.set_zlabel('Action')
    plt.show()
    ax.figure.savefig(f'{algorithm_name}_3D_chart.png')
    
def draw_2D_chart(chart_data_time, chart_data_reward, algorithm_name):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(chart_data_time, chart_data_reward, c='r', marker='o')
    ax.set_xlabel('Time')
    ax.set_ylabel('Reward')
    plt.show()
    ax.figure.savefig(f'{algorithm_name}_2D_chart.png')

def flipcoin(epsilon):
    r = random.random()
    return r < epsilon

def discretize_state(state):
    discrete_state = (min(2, max(-2, int((state[0]) / 0.05))), \
                        min(2, max(-2, int((state[1]) / 0.1))), \
                        min(2, max(-2, int((state[2]) / 0.1))), \
                        min(2, max(-2, int((state[3]) / 0.1))), \
                        min(2, max(-2, int((state[4]) / 0.1))), \
                        min(2, max(-2, int((state[5]) / 0.1))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state

def save_qtable(qtable):
    with open('SavedQtable.txt' , 'w') as f:
        f.write(str(qtable)) 
        
def load_qtable():
    with open('SavedQtable.txt' , 'r') as f:
        return eval(f.read())
    
def save_approximate_weights(weights):
    with open('SavedApproximateWeights.txt', 'w') as f:
        f.write(str(weights))
        
def load_approximate_weights():
    with open('SavedApproximateWeights.txt', 'r') as f:
        return eval(f.read())


class Qtabel(dict):

    def __getitem__(self, idx):
        self.setdefault(idx , 0)
        return dict.__getitem__(self , idx)


class Qlearning :

    def __init__(self , niteration, flag = False) :
        self.flag = flag
        self.niteration = niteration
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = None
        
        self.QValues = Qtabel()
        if flag == True : 
            self.QValues = load_qtable()
        self.action = [0 , 1, 2, 3]

    def getQValue(self , state , action):
        try:
            return self.QValues[(state , action)]
        except KeyError:
            return 0

    def computeValueFromQValues(self, state):
        max = self.getQValue(state , self.action[0])
        for act in self.action :
            if self.getQValue(state , act) > max :
                max = self.getQValue(state , act)
        return max

    def computeActionFromQValues(self, state):
        max = self.computeValueFromQValues(state)
        for act in self.action : 
            if self.getQValue(state , act) == max:
                return act
    
    def getAction(self , state , iternumber):
        self.setparametr(iternumber)
        if flipcoin(self.epsilon):
            action = random.choice(self.action)
        else:
            action = self.computeActionFromQValues(state)
        return action

    def update(self , state , action , nextstate , reward , iternumber ,terminated , truncated) :
        if terminated or truncated:
            self.QValues[(state , action)] = self.getQValue(state , action) +  self.alpha * (reward - self.getQValue(state , action))
        else:
            self.QValues[(state , action)] = self.getQValue(state , action) + self.alpha * (reward + (self.gamma *self.computeValueFromQValues(nextstate)) - self.getQValue(state , action))

    def setparametr(self , iternumber):
        if self.flag:
            self.epsilon = 0
        else:
            self.epsilon = (self.niteration - iternumber) / self.niteration

    def train(self , env):
        saved_time, saved_rewards, saved_actions = list(), list(), list()
        nextstate, info = env.reset(seed=42)
        totalreward = 0
        for i in range(self.niteration):
            while True:
                currentstate = nextstate
                currentstate = discretize_state(currentstate)   
                action = self.getAction(currentstate , i + 1)
                nextstate, reward, terminated, truncated, info = env.step(action)
                totalreward += reward
                nextstatetmp = discretize_state(nextstate)
                self.update(currentstate , action , nextstatetmp , reward , i , terminated , truncated)
                env.render()
                if terminated or truncated:
                    saved_time.append(i)
                    saved_rewards.append(totalreward)
                    saved_actions.append(action)
                    print(totalreward)
                    if i % 100 == 0:
                        print("step :" , i)
                    totalreward = 0
                    nextstate, info = env.reset()
                    break
                
            # if i == self.niteration - 1:
            #     save_qtable(self.QValues)
            #     print("Qtable saved in the file.")
                # break
                
        draw_3D_chart(saved_time, saved_rewards, saved_actions, "QLearning")
        draw_2D_chart(saved_time, saved_rewards, "QLearning")
                

class SarsaQlearning(Qlearning):

    def update(self , state , action , nextstate  , nextaction , reward , iternumber ,terminated , truncated) :
        if terminated or truncated:
            self.QValues[(state , action)] = self.getQValue(state , action) + self.alpha * (reward - self.getQValue(state , action))
        else:
            self.QValues[(state , action)] = self.getQValue(state , action) + self.alpha * (reward + (self.gamma *self.getQValue(nextstate , nextaction)) - self.getQValue(state , action))

    def train(self , env ):
        saved_time, saved_rewards, saved_actions = list(), list(), list()
        nextstate, info = env.reset(seed=42)
        totalreward = 0
        for i in range(self.niteration):
            while True:
                currentstate = nextstate
                currentstate = discretize_state(currentstate)   
                action = self.getAction(currentstate , i)
                nextstate, reward, terminated, truncated, info = env.step(action)
                totalreward += reward
                nextstatetmp = discretize_state(nextstate)
                nextaction = self.getAction(nextstatetmp , i)
                self.update(currentstate , action , nextstatetmp  , nextaction, reward , i , terminated , truncated)
                env.render()
                if terminated or truncated:
                    saved_time.append(i)
                    saved_rewards.append(totalreward)
                    saved_actions.append(action)
                    print(totalreward)
                    if i % 100 == 0:
                        print("step :" , i)
                    totalreward = 0
                    nextstate, info = env.reset()
                    break
                
            # if i == self.niteration - 1:
            #     save_qtable(self.QValues)
            #     print(f"Qtable saved in the file. iteration number == {i}")
                # break

        draw_3D_chart(saved_time, saved_rewards, saved_actions, "SarsaQLearning")
        draw_2D_chart(saved_time, saved_rewards, "SarsaQLearning")


class approximateQlearning(Qlearning):

    def __init__(self, niterations, flag=False) -> None:
        self.weights = [1 , 1, 1, 1, 1, 1, 1, 1 , 1]
        if flag == True:
            self.weights = load_approximate_weights()
        Qlearning.__init__(self, niterations, flag)

    def getQValue(self, state, action): 
        q = 0
        for i in range(len(state)):
            q += self.weights[i] * state[i]
        q += self.weights[-1] * action
        return q


    def update(self, state, action, nextstate, reward, iternumber ,terminated , truncated):
        if terminated or truncated:
            diff = reward - self.getQValue(state , action)
        else:
            diff = reward + self.computeValueFromQValues(nextstate) - self.getQValue(state , action)
        
        for i in range(len(state)):
            self.weights[i] += self.alpha * diff * state[i]
            
        self.weights[-1] += self.alpha * diff * action

    def train(self , env):
        nextstate, info = env.reset(seed=42)
        totalreward = 0
        for i in range(self.niteration):
            while True:
                currentstate = nextstate
                action = self.getAction(currentstate , i)
                nextstate, reward, terminated, truncated, info = env.step(action)
                totalreward += reward
                self.update(currentstate , action , nextstate, reward , i , terminated , truncated)
                env.render()
                if terminated or truncated:
                    print(totalreward)
                    if i % 100 == 0:
                        print("step :" , i)
                    totalreward = 0
                    nextstate, info = env.reset()
                    break
                
            if i == self.niteration - 1:
                save_approximate_weights(self.weights)
                print("Weights saved in the file.")

