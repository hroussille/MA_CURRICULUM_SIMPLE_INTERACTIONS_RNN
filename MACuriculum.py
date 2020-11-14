
from MADDPG import MADDPG
import BCActor
import PPO
from tqdm import tqdm
from tqdm import trange
from numpy_ringbuffer import RingBuffer

import numpy as np
import random
import torch
import utils
import copy
import time

def states_callback(x):
    seq_sizes = [seq.shape[0] for seq in x]
    padded = torch.nn.utils.rnn.pad_sequence(x)
    packed = torch.nn.utils.rnn.pack_padded_sequence(padded, seq_sizes, enforce_sorted=False)
    return packed

class MACuriculum():

    def __init__(self, env, writer, run_id, config, path):

        self.env = env
        self.config = config
        self.env_name = config['env']['env_name']
        self.self_play_gamma = config['self_play']['self_play_gamma']
        self.shuffle_self_play = config['self_play']['shuffle']
        self.shuffle_target_play = config['target_play']['shuffle']
        self.writer = writer
        self.run_id = run_id
        self.path = path

        self.learners = MADDPG(env, config['env']['env_obs_learners'], **config['learners'])
        self.teachers = MADDPG(env, config['env']['env_obs_teachers'], **config['teachers'])

        self.stop = PPO.PPO(**config['stop'], states_callback=states_callback)

    def get_teachers_subpolicies(self):
        return [random.randrange(0, self.config['teachers']['subpolicies']) for _ in range(self.env.n)]

    def get_learners_subpolicies(self):
        return [random.randrange(0, self.config['learners']['subpolicies']) for _ in range(self.env.n)]

    def apply_noise_decay(self):
        self.learners.apply_noise_decay()
        self.teachers.apply_noise_decay()

    def run(self):
        target_play_mean = []
        target_play_std = []

        self_play_mean = []
        self_play_std = []
        current_best = 0

        max_episodes = self.config['self_play']['episodes']
        max_timestep = self.config['self_play']['max_timestep']
        max_timestep_alice = self.config['self_play']['max_timestep_alice']
        max_timestep_bob = self.config['self_play']['max_timestep_bob']
        max_exploration_episodes = self.config['self_play']['exploration_episodes']
        stop_probability = self.config['self_play']['exploration_stop_probability']
        tolerance = self.config['self_play']['tolerance']
        stop_update = self.config['self_play']['stop_update_freq']
        set_update = self.config['self_play']['set_update_freq']
        set2_update = self.config['self_play']['set2_update_freq']
        mode = self.config['self_play']['mode']
        alternate = self.config['self_play']['alternate']
        alternate_step = self.config['self_play']['alternate_step']
        test_freq = self.config['self_play']['test_freq']
        test_episodes = self.config['self_play']['test_episodes']
        max_timestep_target = self.config['target_play']['max_timestep']
        max_episodes_target = self.config['target_play']['episodes']
        max_exploration_episodes_target = self.config['target_play']['exploration_episodes']
        test_freq_target = self.config['target_play']['test_freq']
        test_episodes_target = self.config['target_play']['test_episodes']

        max_timestep_strategy = self.config['self_play']['max_timestep_strategy']
        ma_window_length = self.config['self_play']['ma_window_length']
        ma_multiplier = self.config['self_play']['ma_multiplier']
        ma_default_value = self.config['self_play']['ma_default_value']
        ma_bias = self.config['self_play']['ma_bias']

        #increment = 1. / max_episodes
        increment = 1. / 100000

        t = trange(max_exploration_episodes, desc='Self play exploration', leave=True)

        for episode in t:
            t.set_description("Self play exploration")
            t.refresh()
            eval('self.explore_self_play_{}(max_timestep, tolerance, stop_probability)'.format(mode))

        t = trange(max_episodes, desc='Self play training', leave=True)
        train_teacher = True
        last_switch = 0

        if max_timestep_strategy == "auto":
            time_buffer = RingBuffer(capacity=ma_window_length)

            for _ in range(ma_window_length):
                time_buffer.append(ma_default_value)

            max_timestep = int(np.ceil(ma_multiplier * np.mean(time_buffer)))

        for episode in t:
            t.set_description("Self play training")
            t.refresh()

            tA, tB = eval('self.self_play_{}(max_timestep_alice, max_timestep_bob, episode, tolerance, stop_update, set_update, alternate, train_teacher)'.format(mode))

            if max_timestep_strategy == "auto":
                time_buffer.append(tA)
                max_timestep = min(int(np.ceil(ma_multiplier * np.mean(time_buffer) + ma_bias)), max_timestep_target)

            if alternate:
                if episode - last_switch >= alternate_step:
                    train_teacher = not(train_teacher)
                    last_switch = episode

            self.apply_noise_decay()
            self.learners.increment_per(increment)
            self.teachers.increment_per(increment)

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes, max_timestep_target, tolerance, render=False)
                self.writer.add_scalars("self_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                self_play_mean.append(test_mean)
                self_play_std.append(test_std)

                if test_mean >= current_best:
                    current_best = test_mean
                    self.learners.save(self.path + "/models_{}".format(self.run_id))

        return self_play_mean, self_play_std

        self.learners.clear_rb()

        t = trange(max_exploration_episodes_target, desc='Target play exploration', leave=True)

        for episode in t:
            t.set_description("Target play exploration")
            t.refresh()
            self.explore_target_play(max_timestep_target, tolerance)

        t = trange(max_episodes_target, desc='Target play training', leave = True)
        for episode in t:
            t.set_description("Target play training")
            t.refresh()
            self.target_play(max_timestep_target, episode, tolerance)

            if episode % test_freq == 0:
                test_mean, test_std = self.test(test_episodes_target, max_timestep_target, tolerance, render=False)
                self.writer.add_scalars("Target_play/{}".format(self.run_id), {'average reward': np.mean(test_mean)}, episode)
                target_play_mean.append(test_mean)
                target_play_std.append(test_std)

        return target_play_mean, target_play_std

    def explore_self_play_reverse(self, tMAX, tolerance, set_probability=0.5):

        tA = 0
        tB = 0
        solved = False

        seed = random.randint(0, 2 ** 32 - 1)
        np.random.seed(seed)

        """ Random sampling of finish zone position """
        finish_zone = np.random.uniform(-1, 1, (1, 2))

        """ Random sampling of agents starting pos inside finish_zone"""
        init_pos = np.tile(finish_zone, (self.env.n_agents, 1)) + random.uniform(-0.3, 0.3, (self.env.n_agents, 2))

        subs_teacher = self.get_teachers_subpolicies()
        subs_learners = self.get_learners_subpolicies()

        s = self.env.reset(agents_positions=init_pos, finish_zone_position=finish_zone)
        phase = 0

        landmarks = np.random.uniform(-1, 1, (self.env.n_agents, 2))
        landmarks_flags = np.ones(self.env.n_agents)

        s = utils.state_to_teacher_state(s, landmarks, landmarks_flags)
        s = utils.add_phase_to_state(s, phase)

        while True:

            pass


    def explore_self_play_repeat(self, tMAX, tolerance, set_probability=0.5, stop_probability=0.5):

        tA = 0
        tB = 0
        solved = False

        seed = random.randint(0, 2 ** 32 - 1)
        np.random.seed(seed)
        phase = 0

        s = self.env.reset()

        landmarks = np.random.uniform(-1, 1, (self.env.n_agents, 2))
        landmarks_flags = np.ones(self.env.n_agents)

        s = utils.state_to_teacher_state(s, landmarks, landmarks_flags)
        s = utils.add_phase_to_state(s, phase)

        s_init = copy.deepcopy(s)

        subs_learner = self.get_learners_subpolicies()
        subs_teacher = self.get_teachers_subpolicies()

        teacher_state = {}
        learner_state = {}

        stop_flag = False
        set_flag = False

        while True:

            tA = tA + 1

            if not set_flag:

                set_flag = np.random.rand() < set_probability

                if tA >= tMAX:
                    set_flag = True

                if set_flag:
                    landmarks = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])
                    landmarks_flags = np.zeros(landmarks_flags.shape)
                    phase = 1

            actions_detached = self.teachers.random_act()
            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.state_to_teacher_state(s_t, landmarks, landmarks_flags)
            s_t = utils.add_phase_to_state(s_t, phase)


            stop_flag = np.random.rand() < stop_probability

            if tA >= tMAX:
                stop_flag = True

            if stop_flag or tA >= tMAX:

                finish_zone, finish_zone_radius = utils.compute_finish_zone(np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents]))

                teacher_state['s'] = copy.deepcopy(s)
                teacher_state['s_t'] = copy.deepcopy(s_t)
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True
                s = s_t
                break

            obs = np.hstack((np.array(s_init), np.array(s)))

            obs_t = np.hstack((np.array(s_init), np.array(s_t)))


            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)
            s = s_t

        s_final = copy.deepcopy(s_t)
        np.random.seed(seed)

        s = self.env.reset(landmark_positions=landmarks, finish_zone_position=finish_zone, finish_zone_radius=finish_zone_radius)

        save_s = None
        save_s_t = None

        while True:

            tB = tB + 1
            actions_detached = self.learners.random_act()
            s_t, _, solved, _ = self.env.step(copy.deepcopy(actions_detached))

            if tA + tB >= tMAX or solved:
                learner_state['s'] = copy.deepcopy(s)
                learner_state['s_t'] = copy.deepcopy(s_t)
                learner_state['a'] = copy.deepcopy(actions_detached)
                learner_state['d'] = solved
                break

            reward = 0

            self.learners.push_sample(s, actions_detached, [0] * self.env.n, solved, s_t, subs_learner)

            s = s_t

        if solved is False:
            tB = tMAX - tA

        R_A = [self.self_play_gamma * max(0, tB - tA)] * self.env.n
        R_B = [self.self_play_gamma * -1 * tB] * self.env.n

        obs = np.hstack((np.array(s_init), np.array(teacher_state['s'])))
        obs_t = np.hstack((np.array(s_init), np.array(teacher_state['s_t'])))

        self.teachers.push_sample(obs, teacher_state['a'], R_A, teacher_state['d'], obs_t, subs_teacher)
        self.learners.push_sample(learner_state['s'], learner_state['a'], R_B, solved, learner_state['s_t'], subs_learner)

    def explore_target_play(self, max_timestep, tolerance):

        step_count = 0
        Done = False
        timestep = 0

        s = self.env.reset()

        while timestep < max_timestep and not(Done):
            subs = self.get_learners_subpolicies()
            timestep = timestep + 1
            actions_detached = self.learners.act(s, subs)
            s_t, r, Done, i = self.env.step(copy.deepcopy(actions_detached))

            if timestep >= max_timestep:
                Done = True

            self.learners.push_sample(s, actions_detached, r, Done, s_t, subs)

            s = s_t

    """
        IF BASES_SET IS FALSE : STOP IS INVALID
        IF BASES_SET IS TRUE : SET_BASES IS INVALID
    """
    def get_mask(self, bases_set):
        if bases_set:
            return np.array([True, False, True])
        else:
            return np.array([True, True, True])

    def self_play_repeat(self, max_timestep_alice, max_timestep_bob, episode, tolerance, stop_update, set_update, alternate, train_teacher):
        tA = 0
        tB = 0
        tSet = 0

        seed = random.randint(0, 2 ** 32 - 1)

        np.random.seed(seed)

        phase = 0

        s = self.env.reset()

        landmarks = np.random.uniform(-1, 1, (self.env.n_agents, 2))
        landmarks_flags = np.ones(self.env.n_agents)

        s = utils.state_to_teacher_state(s, landmarks, landmarks_flags)
        s = utils.add_phase_to_state(s, phase)
        s_init = copy.deepcopy(s)

        subs_learner = self.get_learners_subpolicies()
        subs_teacher = self.get_teachers_subpolicies()
        teacher_state = {}
        learner_state = {}

        hidden_actor = None
        hidden_critic = None

        while True:

            tA = tA + 1

            input = np.hstack((np.array(s_init), np.array(s)))
            input_t = torch.Tensor(input)

            actions_detached = self.teachers.act(input_t, subs_teacher)

            s_t, r, done, i = self.env.step(copy.deepcopy(actions_detached))
            s_t = utils.state_to_teacher_state(s_t, landmarks, landmarks_flags)
            s_t = utils.add_phase_to_state(s_t, phase)

            """
                ALWAYS REQUEST STOP CONTROLLER FIRST WITH CURRENT ACTION MASK
            """
            mask = self.get_mask(phase)
            action, log_prob, value, hidden_actor, hidden_critic = self.stop.act(input_t.flatten(), hidden_actor=hidden_actor, hidden_critic=hidden_critic, mask=torch.Tensor(mask))
            action_item = action.item()

            self.stop.memory.current_seq.append(input.flatten())
            self.stop.memory.log_prob.append(log_prob)
            self.stop.memory.actions.append(action)
            self.stop.memory.values.append(value)
            self.stop.memory.masks.append(mask)

            """
                IF ACTION IS 0 : JUST LET THE CONTROLLERS MOVE ON NEXT STEP
                OTHERWISE : HANDLE ACTION AND GENERATE SCENARIO ACCORDINGLY
                
                double check on bases_set should not be necessary thanks to action mask, but we never know...
                second check on tA ensures a fully defined environment when control is passed to BOB
            """
            if action_item == 1 and phase == 0:
                landmarks = np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents])
                landmarks_flags = np.zeros(landmarks_flags.shape)

                tSet = tA
                phase = 1

            if action_item == 2 or tA >= max_timestep_alice:
                finish_zone, finish_zone_radius = utils.compute_finish_zone(np.array([copy.deepcopy(agent.get_pos()) for agent in self.env.agents]))

                teacher_state['s'] = copy.deepcopy(np.hstack((np.array(s_init), np.array(s))))
                teacher_state['s_t'] = copy.deepcopy(np.hstack((np.array(s_init), np.array(s_t))))
                teacher_state['a'] = copy.deepcopy(actions_detached)
                teacher_state['d'] = True

                break

            self.stop.memory.rewards.append(0)
            self.stop.memory.dones.append(False)

            obs = np.hstack((np.array(s_init), np.array(s)))

            obs_t = np.hstack((np.array(s_init), np.array(s_t)))

            self.teachers.push_sample(obs, actions_detached, [0] * self.env.n, False, obs_t, subs_teacher)
            self.teachers.train(subs_learner)

            s = s_t

        np.random.seed(seed)

        s = self.env.reset(landmark_positions=landmarks, landmark_flags=landmarks_flags, finish_zone_position=finish_zone, finish_zone_radius=finish_zone_radius)

        while True:

            tB = tB + 1

            actions_detached = self.learners.act(s, subs_learner)

            s_t, _, solved, _ = self.env.step(copy.deepcopy(actions_detached))

            if tA + tB >= max_timestep_bob or solved:
                learner_state['s'] = copy.deepcopy(s)
                learner_state['s_t'] = copy.deepcopy(s_t)
                learner_state['a'] = copy.deepcopy(actions_detached)
                learner_state['d'] = solved
                break

            self.learners.push_sample(s, actions_detached, [0] * self.env.n, False, s_t, subs_learner)
            self.learners.train(subs_teacher)

            s = s_t

        if not solved:
            tB = max_timestep_bob - tA

        R_A = [self.self_play_gamma * max(0, tB - tA)] * self.env.n
        R_B = [self.self_play_gamma * -1 * tB] * self.env.n

        self.teachers.push_sample(teacher_state['s'], teacher_state['a'], R_A, teacher_state['d'], teacher_state['s_t'], subs_teacher)
        self.learners.push_sample(learner_state['s'], learner_state['a'], R_B, learner_state['d'], learner_state['s_t'], subs_learner)

        self.stop.memory.rewards.append(R_A[0])
        self.stop.memory.dones.append(True)
        self.stop.memory.new_seq()

        nb_bases = np.array([landmark.get_activated() for landmark in self.env.landmarks]).astype(int).sum()

        self.writer.add_scalars("Self play BOB bases activated {}".format(self.run_id), {'Bases activated' : nb_bases}, episode)
        self.writer.add_scalars("Self play episode time {}".format(self.run_id), {'ALICE TIME': tA, 'BOB TIME': tB, 'SET TIME':tSet}, episode)
        self.writer.add_scalars("Self play rewards {}".format(self.run_id), {"ALICE REWARD" : R_A[0], 'BOB REWARD': R_B[0]}, episode)
        self.writer.add_scalars("Self play finish zone radius {}".format(self.run_id), {"FINISH ZONE RADIUS": finish_zone_radius}, episode)

        print("TA : {} TB : {} TS : {} RA : {} RB {} {}".format(tA, tB, tSet, R_A, R_B, "SOLVED" if solved else ""))

        if episode % stop_update == 0:
            self.stop.update()

        return tA, tB

    def target_play(self, max_timestep, episode, tolerance):

        timestep = 1
        total_reward = 0

        subs = self.get_learners_subpolicies()
        s = self.env.reset()
        Done = False

        while timestep <= max_timestep and not(Done):

            timestep = timestep + 1

            input = s
            actions_detached = self.learners.act(input, subs)
            s_t, r, Done , _ = self.env.step(copy.deepcopy(actions_detached))

            total_reward += np.mean(r)

            self.learners.push_sample(s, actions_detached, r, Done, s_t, subs)
            self.learners.train(subs)

            s = s_t

        self.writer.add_scalars('Target play reward {}'.format(self.run_id), {'Reward': total_reward}, episode)

    def test(self, n_episodes, max_episode_timestep, tolerance, render=False):
        results = []

        for episode in range(n_episodes):

            episode_reward = 0

            done = False
            timestep = 0
            subs = self.get_learners_subpolicies()

            s = self.env.reset()

            while timestep < max_episode_timestep and not done:
                timestep = timestep + 1
                actions = []

                actions_detached = self.learners.act(s, subs, noise=False)
                s_t, r, done, _ = self.env.step(copy.deepcopy(actions_detached))

                if timestep >= max_episode_timestep:
                    done = True

                if render:
                    time.sleep(0.25)
                    self.env.render(mode="human")

                episode_reward += r[0]

                s = s_t

            results.append(episode_reward)

        return np.mean(results, axis=0) , np.std(results, axis=0)

