import torch
import torch.distributions
import copy
import numpy as np
import time

class Actor(torch.nn.Module):

    def __init__(self, input_dim, hidden_state_dim, output_dim, recurrent_layers=1):
        super(Actor, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.recurrent_layers = recurrent_layers
        self.hidden_state = self.set_init_state(1)
        self.lstm = torch.nn.LSTM(input_dim, hidden_state_dim, num_layers=recurrent_layers)
        self.network = torch.nn.Sequential(torch.nn.Linear(hidden_state_dim, 128),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(128, output_dim))

    def set_init_state(self, batch_size=1):
        return torch.zeros(batch_size, self.recurrent_layers, self.hidden_state_dim)

    def forward(self, states, hidden_state=None, masks=None):

        if not isinstance(states, torch.nn.utils.rnn.PackedSequence):
            states = states.reshape(1, 1, states.shape[0])

        if hidden_state is not None:
            output, (h_n, c_n) = self.lstm(states, hidden_state)
        else:
            output, (h_n, c_n) = self.lstm(states)

        output = self.network(output)

        if masks is not None:
            masks = masks.type(torch.BoolTensor)
            output = torch.where(masks, output, torch.tensor(-1e+8))

        output = torch.nn.functional.softmax(output, dim=-1)

        dist = torch.distributions.Categorical(output)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, (h_n, c_n)

    def policy(self, states, actions, hidden_state=None, masks=None):

        if hidden_state is None:
            hidden_state = self.set_init_state()

        output, (h_n, c_n) = self.lstm(states)

        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

            features = []

            for i in range(len(unpacked_len)):
                for j in range(unpacked_len[i]):
                    features.append(output[j][i])

            output = torch.stack(features)
        output = self.network(output).squeeze()

        if masks is not None:
            masks = masks.type(torch.BoolTensor)
            output = torch.where(masks, output, torch.tensor(-1e+8))

        output = torch.nn.functional.softmax(output, dim=-1)

        dist = torch.distributions.Categorical(output)

        return dist.log_prob(actions), dist.entropy()


class Critic (torch.nn.Module):

    def __init__(self, input_dim, hidden_state_dim, recurrent_layers=1):
        super(Critic, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.recurrent_layers = recurrent_layers
        self.hidden_state = self.set_init_state(1)
        self.lstm = torch.nn.LSTM(input_dim, hidden_state_dim, num_layers=recurrent_layers)
        self.network = torch.nn.Sequential(torch.nn.Linear(hidden_state_dim, 128),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(128, 128),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.Linear(128, 1))

    def set_init_state(self, batch_size=1):
        return torch.zeros(batch_size, self.recurrent_layers, self.hidden_state_dim)

    def forward(self, states, hidden_state=None):

        if not isinstance(states, torch.nn.utils.rnn.PackedSequence):
            states = states.reshape(1, 1, states.shape[0])

        if hidden_state is not None:
            output, (h_n, c_n) = self.lstm(states, hidden_state)
        else:
            output, (h_n, c_n) = self.lstm(states)

        if isinstance(output, torch.nn.utils.rnn.PackedSequence):
            output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)

            features = []

            for i in range(len(unpacked_len)):
                for j in range(unpacked_len[i]):
                    features.append(output[j][i])

            output = torch.stack(features)

        return self.network(output).squeeze(), (h_n, c_n)

class Memory():

    def __init__(self, states_callback=None):

        self.last_episode_index = 0
        self.current_seq = []

        self.__reset_arrays()

        if states_callback is None:
            self.states_callback = lambda x: torch.Tensor(x)
        else:
            self.states_callback = states_callback

    def __reset_arrays(self):
        self.states = []
        self.actions = []
        self.log_prob = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []

    def new_seq(self):
        self.states.append(torch.Tensor(self.current_seq).reshape(-1, self.current_seq[0].shape[0]))
        self.current_seq = []

    def push(self, state, action, log_prob, reward, done, value, mask):
        self.current_seq.append(copy.deepcopy(state))
        self.actions.append(copy.deepcopy(action))
        self.log_prob.append(copy.deepcopy(log_prob))
        self.rewards.append(copy.deepcopy(reward))
        self.dones.append(copy.deepcopy(done))
        self.values.append(copy.deepcopy(value))
        self.masks.append(copy.deepcopy(mask))

    def sample(self, batch_size):

        if batch_size == "rollout":
            indices = np.arange(0, len(self.states))
        else:
            indices = np.random.randint(0, len(self.states), batch_size)

        states = self.states_callback(self.states)
        actions = torch.Tensor(self.actions).flatten().detach()
        log_prob = torch.Tensor(self.log_prob).flatten().detach()
        rewards = torch.Tensor(self.rewards).flatten().detach()
        dones = torch.Tensor(self.dones).flatten().detach()
        values = torch.Tensor(self.values).flatten().detach()
        rewards_to_go = torch.Tensor(self.rewards_to_go).flatten().detach()
        advantages = torch.Tensor(self.advantages).flatten().detach()
        masks = torch.Tensor(self.masks).detach()

        return states, actions, log_prob, rewards, dones, values, rewards_to_go, advantages, masks

    def compute_advantages(self, gamma, tau, use_gae=True):
        self.rewards_to_go = self.compute_rewards_to_go(self.rewards, self.dones, gamma)

        if use_gae:
            self.advantages = self.gae(self.rewards, self.dones, self.values, gamma, tau)

        #self.states = np.array(self.states)
        self.actions = np.array(self.actions)
        self.log_prob = np.array(self.log_prob)
        self.rewards = np.array(self.rewards)
        self.dones = np.array(self.dones)
        self.values = np.array(self.values)
        self.rewards_to_go = np.array(self.rewards_to_go)
        self.masks = np.array(self.masks)

        if not use_gae:
            self.advantages = self.rewards_to_go - self.values

        self.advantages = np.array(self.advantages)

    def gae(self, rewards, dones, values, gamma, tau):

        previous_value = 0
        previous_advantage = 0

        deltas = np.zeros(len(rewards))
        advantages = np.zeros(len(rewards))

        for i in reversed(range(len(rewards))):

            if dones[i]:
                previous_value = 0
                previous_advantage = 0

            deltas[i] = rewards[i] + gamma * previous_value * (1 - dones[i]) - values[i]
            advantages[i] = deltas[i] + gamma * tau * previous_advantage * (1 - dones[i])

            previous_value = values[i]
            previous_advantage = advantages[i]

        return advantages

    def compute_rewards_to_go(self, rewards, dones, gamma):

        discounted_rewards = []
        discounted_reward = 0

        for reward, is_terminal in zip(reversed(rewards), reversed(dones)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (gamma * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def __len__(self):
        return len(self.states)

    def clear(self):
        self.__reset_arrays()

class PPO():

    def __init__(self, state_dim, action_dim, hidden_state_dim=64, lr_actor=1e-3, lr_actor_rnn=1e-3, lr_critic=1e-3, lr_critic_rnn=1e-3, clip=0.1, gamma=0.95, tau=0.95, batch_size=64, target_kl=0.01, weight_decay=0.0, continuous=False, normalize_advantage=True, use_gae=True, K_policy=12, K_value=12, entropy_factor=0.01, states_callback=None, vclip=None, update_step=128):

        self.clip = clip
        self.gamma = gamma
        self.tau = tau
        self.K_policy = K_policy
        self.K_value = K_value
        self.entropy_factor = entropy_factor
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.weight_decay = weight_decay
        self.continuous = continuous
        self.batch_size = batch_size
        self.vclip = vclip
        self.use_gae = use_gae
        self.update_step = update_step

        self.actor = Actor(state_dim, hidden_state_dim, action_dim)
        self.actor_old = Actor(state_dim, hidden_state_dim, action_dim)

        self.critic = Critic(state_dim, hidden_state_dim)

        #self.actor_optimizer = torch.optim.RMSprop(self.actor.parameters(), lr_actor, weight_decay=weight_decay)
        #self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr_critic, weight_decay=weight_decay)

        self.actor_optimizer = torch.optim.RMSprop([
            {'params': self.actor.lstm.parameters(), 'lr': lr_actor_rnn},
            {'params': self.actor.network.parameters(), 'lr': lr_actor}]
            , weight_decay=weight_decay)

        self.critic_optimizer = torch.optim.RMSprop([
            {'params': self.critic.lstm.parameters(), 'lr': lr_critic_rnn},
            {'params': self.critic.network.parameters(), 'lr': lr_critic}]
            , weight_decay=weight_decay)

        self.memory = Memory(states_callback=states_callback)

        self.MSELoss = torch.nn.MSELoss()

    def memory_push(self, state, action, log_prob, reward, done, value, mask=None):

        if not torch.is_tensor(action):
            action = torch.Tensor(action)

        if not torch.is_tensor(log_prob):
            log_prob = torch.Tensor(log_prob)

        self.memory.push(state, action, log_prob, reward, done, value, mask)

    def memory_clean(self):
        self.memory.clear()

    def act(self, state, hidden_actor=None, hidden_critic=None, mask=None):
        with torch.no_grad():
            action, log_prob, hidden_actor = self.actor_old(state, hidden_actor, mask)
            value, hidden_critic = self.critic(state, hidden_critic)

        return action.squeeze(), log_prob.squeeze(), value.squeeze(), hidden_actor, hidden_critic

    def update(self):

        self.memory.compute_advantages(self.gamma, self.tau, use_gae=self.use_gae)

        for _ in range(self.K_policy):

            states, old_actions, old_log_probs, rewards, dones, values, rewards_to_go, advantages, masks = self.memory.sample(self.batch_size)

            if self.normalize_advantage:
                advantages = (advantages - advantages.mean()) / advantages.std()

            log_probs, entropy = self.actor.policy(states, old_actions, masks)
            log_probs = log_probs.flatten()

            self.actor_optimizer.zero_grad()

            ratios = torch.exp(log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages

            loss_actor = - (torch.min(surr1, surr2).mean() + self.entropy_factor * entropy.mean())
            loss_actor.backward()
            self.actor_optimizer.step()

            if (old_log_probs - log_probs).mean() > 1.5 * self.target_kl:
                break

        for _ in range(self.K_value):

            states, _, _, _, _, _, rewards_to_go, _, _ = self.memory.sample(self.batch_size)

            values, _ = self.critic(states)
            self.critic_optimizer.zero_grad()
            loss_critic = self.MSELoss(values.flatten(), rewards_to_go.flatten())
            loss_critic.backward()

            if self.vclip is not None:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.vclip)

            self.critic_optimizer.step()

        self.memory.clear()
        self.actor_old.load_state_dict(self.actor.state_dict())
