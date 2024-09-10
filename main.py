import gym
import numpy as np
from rlagents import SarsaQlearning , Qlearning , approximateQlearning 

env = gym.make("LunarLander-v2")
# env = gym.make("LunarLander-v2", render_mode="human")

# agent = Qlearning(10000)
# agent.train(env)

sarsaagent = SarsaQlearning(10000)
sarsaagent.train(env)

# appxagent = approximateQlearning(5)
# appxagent.train(env)

env.close()
