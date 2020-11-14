import numpy as np


class Entity:
    def __init__(self, min=-1, max=1, color=(0.5, 0.5, 0.5), size=0.1, max_speed=1.5):
        self.min = min
        self.max = max
        self.pos = None
        self.speed = None
        self.color = color
        self.size = size
        self.max_speed= max_speed

        self.reset_pos()

    def reset_pos(self):
        self.pos = np.random.uniform(self.min, self.max, 2)
        self.speed = np.zeros(2)

    def get_pos(self):
        return self.pos

    def get_speed(self):
        return self.speed

    def set_pos(self, pos):
        self.pos = pos
        self.speed= np.zeros(2)

    def set_speed(self, speed):
        self.speed = np.clip(self.speed + speed, -self.max_speed, self.max_speed)

    def get_color(self):
        return self.color

    def set_color(self, color):
        self.color = color

    def get_size(self):
        return self.size

    def set_size(self, size):
        self.size = size

    def apply_forces(self, speed, dampening):

        self.speed = self.speed * dampening
        self.speed += speed * 0.1

        """
        speed = np.linalg.norm(self.speed)

        if speed > self.max_speed:
            self.speed = self.speed / speed * self.max_speed
        """

        self.pos += self.speed * 0.1


class Agent(Entity):

    def __init__(self, min=-1, max=1, color=(0, 0, 0.5), size=0.05):
        super().__init__(min, max, color, size)


class Landmark(Entity):

    def __init__(self, min=-1, max=1, color=(0.5 , 0, 0), size=0.1, activated=False):
        super().__init__(min, max, color, size)
        self.activated = activated

    def set_activated(self, activated):
        self.activated = activated

    def get_activated(self):
        return self.activated


class FinishZone(Entity):

    def __init__(self, min=-1, max=1, color=(0.5, 0.5, 0.5), size=0.3):
        super().__init__(min, max, color, size)