import numpy as np
import gym
import Entities

class SimpleInteractions():

    def __init__(self, n_agents=1, with_finish_zone=True, synchronized_activation=True, dampening=0.75):

        self.n_agents = n_agents
        self.n = n_agents

        """ Environment configuration """
        self.with_finish_zone = with_finish_zone
        self.synchronized_activation = synchronized_activation
        self.dampeding = dampening

        """ Build required entities """
        self.agents = [Entities.Agent() for _ in range(self.n_agents)]
        self.landmarks = [Entities.Landmark() for _ in range(self.n_agents)]

        """ Build per agent action space """
        self.action_space = [gym.spaces.Box(-1.0, 1.0, (2,)) for _ in range(self.n_agents)]

        """
            2 : agent position
            2 : agent velocity
            2 * (n_agents - 1) : relative position to other agents
            2 * n_agents : relative position to landmarks
            n_agents : landmark flags
        """
        obs_dim = 2 + 2 + 2 * (n_agents - 1) + 2 * n_agents + n_agents

        """
            If we are using finish zones :
            2 : relative position to finish zone
            1 : finish zone radius
        """
        if with_finish_zone:
            obs_dim = obs_dim + 2 + 1
            self.finish_zone = Entities.FinishZone()

        """ Build per agent observation space """
        self.observation_space = [gym.spaces.Box(low=0, high=0, shape=(obs_dim,)) for _ in range(self.n_agents)]

        self.render_geoms = None
        self.render_geoms_xform = None
        self.viewer = None
        self.rebuild_geoms = False

    def get_entities(self):
        entities = self.agents + self.landmarks

        if self.with_finish_zone:
            entities = entities + [self.finish_zone]

        return entities

    def _is_landmark_activable(self, landmark):
        for agent in self.agents:
            if np.linalg.norm(agent.get_pos() - landmark.get_pos()) <= landmark.get_size():
                return True
        return False

    def _activate_landmarks_synchronous(self):
        for landmark in self.landmarks:
            if not self._is_landmark_activable(landmark):
                self.rebuild_geoms = False
                return

        for landmark in self.landmarks:
            landmark.set_activated(True)
            landmark.set_color([0, 0.5, 0])
            self.rebuild_geoms = True

    def _activate_landmarks_asynchronous(self):
        for landmark in self.landmarks:
            if not landmark.get_activated() and self._is_landmark_activable(landmark):
                landmark.set_activated(True)
                landmark.set_color([0, 0.5, 0])
                self.rebuild_geoms = True

    def activate_landmarks(self):
        if self.synchronized_activation:
            self._activate_landmarks_synchronous()
        else:
            self._activate_landmarks_asynchronous()

    def get_distances_to_finish_zone(self):
        agents_pos = np.array([agent.get_pos() for agent in self.agents])
        finish_zone_pos = np.tile(self.finish_zone.get_pos(), (self.n_agents, 1))

        return np.linalg.norm(agents_pos - finish_zone_pos, axis=1)

    def get_agent_relative_position(self, agent, target):
        return target.get_pos() - agent.get_pos()

    def get_agents_relative_positions_to_agents(self):
        relative_positions = []

        for current in range(self.n_agents):
            relative = []
            for other in range(self.n_agents):
                if current == other:
                    continue
                relative.append(self.get_agent_relative_position(self.agents[current], self.agents[other]))
            relative_positions.append(np.array(relative).flatten())

        return np.array(relative_positions)

    def get_agents_relative_positions_to_landmarks(self):
        relative_positions = []
        for current in range(self.n_agents):
            relative = []
            for other in range(self.n_agents):
                relative.append(self.get_agent_relative_position(self.agents[current], self.landmarks[other]))
            relative_positions.append(np.array(relative).flatten())

        return np.array(relative_positions)

    def get_agents_relative_positions_to_finish_zone(self):
        agents_pos = np.array([agent.get_pos() for agent in self.agents])
        finish_zone_pos = np.tile(self.finish_zone.get_pos(), (self.n_agents, 1))

        return finish_zone_pos - agents_pos

    def get_finish_zone_radius(self):
        return np.array([[self.finish_zone.get_size()] for _ in range(self.n_agents)])

    def get_agents_informations(self):
        return [np.hstack([agent.get_speed(), agent.get_pos()]) for agent in self.agents]

    def get_landmark_flags(self):
        return np.tile([landmark.get_activated() for landmark in self.landmarks], (self.n_agents, 1))

    def get_observations(self):

        obs = []
        obs.append(self.get_agents_informations())
        obs.append(self.get_agents_relative_positions_to_landmarks())
        obs.append(self.get_agents_relative_positions_to_agents())
        obs.append(self.get_landmark_flags())

        if self.with_finish_zone:
            obs.append(self.get_agents_relative_positions_to_finish_zone())
            obs.append(self.get_finish_zone_radius())

        return np.hstack(obs)

    def get_done(self):

        all_landmarks_activated = np.all([landmark.get_activated() for landmark in self.landmarks])

        if not self.with_finish_zone:
            return all_landmarks_activated

        all_agents_in_finish_zone = (self.get_distances_to_finish_zone() <= self.finish_zone.get_size()).all()

        if all_landmarks_activated and all_agents_in_finish_zone:
            return True

        return False

    def step(self, actions):

        actions = np.clip(actions, -1, 1) * 5

        for agent in range(self.n_agents):
            #self.agents[agent].set_speed(actions[agent])
            self.agents[agent].apply_forces(actions[agent], self.dampeding)

        self.activate_landmarks()
        observations = self.get_observations()
        done = self.get_done()
        reward = [1 if done else 0 for _ in range(self.n_agents)]

        return observations, reward, done, None

    def reset_agents(self, agents_positions):

        for index, agent in enumerate(self.agents):
            agent.reset_pos()
            if agents_positions is not None:
                agent.set_pos(agents_positions[index])

    def reset_landmarks(self, landmarks_positions, landmarks_flags):

        for index, landmark in enumerate(self.landmarks):
            landmark.reset_pos()
            landmark.set_activated(False)
            landmark.set_color((0.5, 0, 0))

            if landmarks_positions is not None:
                landmark.set_pos(landmarks_positions[index])

            if landmarks_flags is not None:
                landmark.set_activated(bool(landmarks_flags[index]))

                if landmarks_flags[index]:
                    landmark.set_color([0, 0.5, 0])

    def reset_finish_zone(self, finish_zone_position, finish_zone_radius):
        self.finish_zone.reset_pos()
        self.finish_zone.set_size(0.3)

        if finish_zone_position is not None:
            self.finish_zone.set_pos(finish_zone_position)

        if finish_zone_radius is not None:
            self.finish_zone.set_size(finish_zone_radius)

    def reset(self, agents_positions=None, landmark_positions=None, landmark_flags=None, finish_zone_position=None, finish_zone_radius=None):

        if self.viewer is not None:
            self.viewer.clear_geoms()
            self.render_geoms = None
            self.render_geoms_xform = None

        self.reset_agents(agents_positions)
        self.reset_landmarks(landmark_positions, landmark_flags)

        if self.with_finish_zone:
            self.reset_finish_zone(finish_zone_position, finish_zone_radius)

        self.rebuild_geoms = True
        #self.activate_landmarks()

        return self.get_observations()

    def render(self, mode="human"):

        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(700, 700)

        # create rendering geometry
        if self.render_geoms is None or self.rebuild_geoms:
            import rendering
            self.viewer.clear_geoms()
            # import rendering only if we need it (and don't import for headless machines)
            # from gym.envs.classic_control import rendering
            self.render_geoms = []
            self.render_geoms_xform = []

            for entity in self.get_entities():
                geom = rendering.make_circle(entity.get_size())
                xform = rendering.Transform()
                geom.set_color(*entity.get_color(), alpha=0.5)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for geom in self.render_geoms:
                self.viewer.add_geom(geom)

        results = []
        # update bounds to center around agent
        cam_range = 1
        pos = np.zeros(2)
        self.viewer.set_bounds(pos[0] - cam_range, pos[0] + cam_range, pos[1] - cam_range, pos[1] + cam_range)
        # update geometry positions
        for a, entity in enumerate(self.get_entities()):
            self.render_geoms_xform[a].set_translation(*entity.get_pos())

        # render to display or array
        results.append(self.viewer.render(return_rgb_array=mode == 'rgb_array'))

        return results