import numpy as np
import torch
import random
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from prioritized_replay_buffer import PrioritizedReplayBuffer
from ActorCritic import Actor
from ActorCritic import Critic

writer = None


class MADDPG():

    actors = None
    actors_targets = None
    actors_optimisers = None
    critics = None
    critics_targets = None
    critics_optimisers = None
    env = None
    gamma = None

    def __init__(self, env, env_obs, gamma=0.99, tau=0.001, lr_actor=1e-3, lr_critic=1e-3, weight_decay=0.1, batch_size=64, subpolicies=1, action_shape=2, replay_buffer_size=5000, replay_buffer_type="rb", noise=0.1, noise_decay=0.999, max_action=1, min_action=-1, teacher=False, alpha=0.1, bc=None):

        self.env = env
        self.subpolicies = subpolicies
        self.total_obs = np.sum(env_obs)
        self.weight_decay = weight_decay
        self.env_obs = env_obs
        self.max_action = max_action
        self.min_action = min_action
        self.action_shape = action_shape
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_buffer_type = replay_buffer_type
        self.replay_buffer_size = replay_buffer_size
        self.init_noise = noise
        self.noise = noise
        self.noise_decay = noise_decay
        self.teacher = teacher
        self.bc = bc
        self.alpha = alpha

        self.mul = 1 if self.teacher is False else 2

        self.actors = [[Actor(self.mul * env_obs[agent], action_shape) for i in range(self.subpolicies)] for agent in range(env.n)]
        self.actors_targets = [[Actor(self.mul * env_obs[agent], action_shape) for i in range(self.subpolicies)] for agent in range(env.n)]
        self.critics = [Critic(self.mul * self.total_obs + action_shape * len(env.agents)) for _ in env.agents]
        self.critics_targets = [Critic(self.mul * self.total_obs + action_shape * len(env.agents)) for _ in env.agents]

        self.actors_optimizers = [[torch.optim.RMSprop(self.actors[agent][i].parameters(), lr=lr_actor, weight_decay=weight_decay) for i in range(self.subpolicies)] for agent in range(len(env.agents))]
        self.critics_optimisers = [torch.optim.RMSprop(self.critics[agent].parameters(), lr=lr_critic ,weight_decay=weight_decay) for agent in range(len(env.agents))]

        if self.subpolicies > 1:
            if self.replay_buffer_type == "rb":
                self.replay_buffers = [[ReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in range(env.n)]
            else:
                self.replay_buffers = [[PrioritizedReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in range(env.n)]
        else:
            if self.replay_buffer_type == "rb":
                self.replay_buffers = ReplayBuffer(self.replay_buffer_size)
            else:
                self.replay_buffers = [[PrioritizedReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in range(env.n)]

    def save(self, path):
        for agent in range(self.env.n):
            torch.save(self.critics[agent].state_dict(), path + '/critic_{}.pt'.format(agent))
            for sub in range(self.subpolicies):
                torch.save(self.actors[agent][sub].state_dict(), path + '/actor_{}_subpolicy_{}.pt'.format(agent, sub))

    def load(self, path):
        for agent in range(self.env.n):
            self.critics[agent].load_state_dict(torch.load(path + "/critic_{}.pt".format(agent)))
            for sub in range(self.subpolicies):
                self.actors[agent][sub].load_state_dict(
                    torch.load(path + "/actor_{}_subpolicy_{}.pt".format(agent, sub)))

    def push_sample(self, s, a, r, d, s_t, subs):

        if self.replay_buffer_type == "rb":
            if self.subpolicies > 1:
                for agent in range(self.env.n):
                    self.replay_buffers[agent][subs[agent]].push(s, a, r, d, s_t, subs)
            else:
                self.replay_buffers.push(s, a, r, d, s_t, subs)

        else:
            errors = self.get_errors(s, a, r, d, s_t, subs)

            for agent in range(self.env.n):
                self.replay_buffers[agent][subs[agent]].push(s, a, r, d, s_t, subs, errors[agent])

    def get_errors(self, s, a, r, d, s_t, subs):
        errors = []

        per_agent_obs = [s[agent] for agent in range(self.env.n)]

        t_agent_obs = [torch.Tensor(list(per_agent_obs[agent][:])).view(1, -1) for agent in range(self.env.n)]

        obs = torch.cat(t_agent_obs, 1)

        per_agent_new_obs = [s_t[agent] for agent in range(self.env.n)]
        t_agent_new_obs = [torch.Tensor(list(per_agent_new_obs[agent][:])).view(1, -1) for agent in range(self.env.n)]
        new_obs = torch.cat(t_agent_new_obs, 1)

        action = torch.Tensor(a)
        reward = torch.Tensor(r)

        new_actions = []

        for agent in range(self.env.n):
            current_new_action = self.actors_targets[agent][subs[agent]](t_agent_new_obs[agent]).clamp(self.min_action, self.max_action)
            new_actions.append(current_new_action)

        new_actions = torch.stack(new_actions, 1)

        for i in range(len(self.env.agents)):

                with torch.no_grad():

                    target_Q_input = torch.cat((new_obs, new_actions.view(1, -1)), 1)
                    target_Q = reward[i] + (self.gamma * self.critics_targets[i](target_Q_input) * (1 - d))
                    current_Q_input = torch.cat((obs, action.view(1, -1)), 1)
                    current_Q = self.critics[i](current_Q_input)

                    critic_loss = (current_Q - target_Q) ** 2
                    errors.append(critic_loss.item())

        return errors

    def increment_per(self, increment):
        if self.replay_buffer_type == "per":

            for agent in range(self.env.n):
                for rb in self.replay_buffers[agent]:
                    rb.increment_beta(increment)

    def apply_noise_decay(self):
        self.noise = self.noise * self.noise_decay

    def reset_noise(self, noise_value=None):

        if noise_value is None:
            self.noise = self.init_noise

        else:
            self.noise = noise_value

    def random_act(self):
        return np.array([random.uniform(self.min_action, self.max_action) for _ in range(self.env.n * self.action_shape)]).reshape(self.env.n, self.action_shape)

    def clear_rb(self):

        if self.subpolicies > 1:
            if self.replay_buffer_type == "rb":
                self.replay_buffers = [[ReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in range(self.env.n)]
            else:
                self.replay_buffers = [
                    [PrioritizedReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in range(self.env.n)]
        else:
            if self.replay_buffer_type == "rb":
                self.replay_buffers = ReplayBuffer(self.replay_buffer_size)
            else:
                self.replay_buffers = [
                    [PrioritizedReplayBuffer(self.replay_buffer_size) for _ in range(self.subpolicies)] for _ in
                    range(self.env.n)]

    def act(self, s, subs, noise=True):

        actions = []
        input = torch.Tensor(s)

        for agent in range(self.env.n):
            action = self.actors[agent][subs[agent]](input[agent])

            if noise is True:
                action = action + torch.Tensor(action.shape).uniform_(-self.noise, self.noise)

            action = action.clamp(self.min_action, self.max_action)
            actions.append(action)

        return np.array([action.detach().cpu().numpy() for action in actions]).reshape(self.env.n, self.action_shape)

    def train(self, subs):

        for i in range(len(self.env.agents)):
            self.critics_optimisers[i].zero_grad()

            if self.subpolicies > 1:
                minibatch = self.replay_buffers[i][subs[i]].sample(self.batch_size)
            else:
                if self.replay_buffer_type == "rb":
                    minibatch = self.replay_buffers.sample(self.batch_size)
                else:
                    minibatch = self.replay_buffers[i][subs[i]].sample(self.batch_size)

            if self.replay_buffer_type == "rb":
                obs, actions, rewards, dones, new_obs, _ = minibatch
            else:
                (batch, index, is_weight) = minibatch
                is_weight = torch.Tensor(is_weight)
                obs, actions, rewards, dones, new_obs, _ = batch

            """ Handle heterogeneous observation sizes by splitting agent observations """
            per_agent_obs = [obs[:, agent] for agent in range(self.env.n)]
            t_agent_obs = [torch.Tensor(list(per_agent_obs[agent][:])) for agent in range(self.env.n)]
            obs = torch.cat(t_agent_obs, 1)

            per_agent_new_obs = [new_obs[:, agent] for agent in range(self.env.n)]
            t_agent_new_obs = [torch.Tensor(list(per_agent_new_obs[agent][:])) for agent in range(self.env.n)]
            new_obs = torch.cat(t_agent_new_obs, 1)

            actions = torch.Tensor(actions).clone()
            rewards = torch.Tensor(rewards)
            dones = torch.Tensor(dones).view(self.batch_size, -1)


            """ Compute a(st + 1) for each agent """
            new_actions = []

            for agent in range(self.env.n):
                current_new_action = self.actors_targets[agent][subs[agent]](t_agent_new_obs[agent]).clamp(self.min_action, self.max_action)
                new_actions.append(current_new_action)

            new_actions = torch.stack(new_actions, 1)

            target_Q_input = torch.cat((new_obs, new_actions.view(self.batch_size, -1)), 1)
            current_Q_input = torch.cat((obs, actions.view(self.batch_size, -1)), 1)

            with torch.no_grad():
                target_Q = rewards[:, i].view(self.batch_size, -1) + (self.gamma * self.critics_targets[i](target_Q_input) * (1 - dones))
                target_Q.detach()

            current_Q = self.critics[i](current_Q_input)

            if self.replay_buffer_type == "per":
                error = ((current_Q - target_Q) ** 2).reshape(is_weight.shape)
                error = error.cpu().detach().numpy()

                for sample_index in range(self.batch_size):
                    if self.subpolicies > 1:
                        self.replay_buffers[i][subs[i]].update(index[sample_index], error[sample_index])

            critic_loss = (is_weight * ((current_Q - target_Q) ** 2)).mean()
            critic_loss.backward()

            self.critics_optimisers[i].step()

            self.actors_optimizers[i][subs[i]].zero_grad()

            t_agent_obs[i].requires_grad = True
            actions[:, i] = self.actors[i][subs[i]](t_agent_obs[i])

            _input = torch.cat((obs.view(self.batch_size, -1), actions.view(self.batch_size, -1)), 1)

            actor_loss = - self.critics[i](_input).mean()

            if self.bc is not None:
                bc_loss = self.alpha * F.mse_loss(actions[:, i], self.bc[i](t_agent_obs[i][:, :self.env_obs[i]]))
                actor_loss += bc_loss

            actor_loss.backward()

            self.actors_optimizers[i][subs[i]].step()

            # Update the frozen target models
            for param, target_param in zip(self.critics[i].parameters(), self.critics_targets[i].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actors[i][subs[i]].parameters(), self.actors_targets[i][subs[i]].parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



