from randomAgentCooperativeNavigation import make_env
import numpy as np
import copy
import time

env, scenario, world = make_env('simple_spread')

def test(n_episodes, max_episode_timestep, render=False):

    for episode in range(n_episodes):

        episode_reward = np.zeros(env.n)

        Done = False
        timestep = 0

        s = env.reset()

        while timestep < max_episode_timestep and Done is False:
            timestep = timestep + 1

            actions_detached = np.ones((3, 2)) * 3
            print(actions_detached)
            s_t, r, done, i = env.step(copy.deepcopy(actions_detached))

            Done = True in done

            if render:
                env.render(mode="human")
                time.sleep(5)

            episode_reward += r

            s = s_t

test(10, 8, True)
