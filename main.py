import gymnasium as gym
import numpy as np
from rlagents import SarsaQlearning , Qlearning , approximateQlearning 

# env = gym.make("LunarLander-v3")
env = gym.make("LunarLander-v3", render_mode="human")

# agent = Qlearning(10000)
# agent.train(env)

sarsaagent = SarsaQlearning(10000 , flag=True)
sarsaagent.train(env)

# appxagent = approximateQlearning(5)
# appxagent.train(env)

env.close()
