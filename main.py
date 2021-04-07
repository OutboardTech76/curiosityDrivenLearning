"""
Main module of the Curiosity Driven Learning Project
"""
import retro
from utils import action_helper as ah

### ACTION SET
### X   X   X   X   X   X   X   X   X
### B   0   s   S   U   D   L   R   A
# 0 = NULL
# s = select
# S = start

MARIO_ACTION_MASK =[1,0,0,0,0,1,1,1,1]

if __name__=="__main__":

    env = retro.make(game="SuperMarioBros-Nes")
    obs = env.reset()
    print("Image shape: {}".format(obs.shape))
    while True:
        action_taken = env.action_space.sample()
        # print(ah.full_action_2_partial_action(action_taken,MARIO_ACTION_MASK))
        obs, rew, done, info = env.step(action_taken)
        env.render()
        if done:
            obs = env.reset()
            break
    env.close()
