"""
Credits to https://github.com/jlko/STOVE
"""

import numpy as np
from spriteworld import renderers as spriteworld_renderers
from spriteworld.sprite import Sprite

def norm(x):
    """Overloading numpys default behaviour for norm()."""
    if len(x.shape) == 1:
        _norm = np.linalg.norm(x)
    else:
        _norm = np.linalg.norm(x, axis=1).reshape(-1, 1)
    return _norm

class PhysicsEnv:
    """Base class for the physics environments."""

    def __init__(self, num_objects=3, object_radius=1., object_mass=1., coord_limits=10, granularity=5, size=32, t=1.,
                 init_v_factor=None, friction_coefficient=0., seed=None,
                 sprites=False, use_colors=None):
        """Initialize a physics env with some general parameters.

        Args:
            num_objects (int): Optional, number of objects in the scene.
            object_radius (float)/list(float): Optional, radius of objects in the scene.
            object_mass (float)/list(float): Optional, mass of the objects in the scene.
            coord_limits (float): Optional, coordinate limits of the environment.
            eps (float): Optional, internal simulation granularity as the
                fraction of one time step. Does not change speed of simulation.
            size (int): Optional, pixel resolution of the images.
            t (float): Optional, dt of the step() method. Speeds up or slows
                down the simulation.
            init_v_factor (float): Scaling factor for inital velocity. Used only
                in Gravity Environment.
            friction_coefficient (float): Friction slows down balls.
            seed (int): Set random seed for reproducibility.
            sprites (bool): Render selection of sprites using spriteworld
                instead of balls.

        """
        np.random.seed(seed)

        self.n = num_objects
        self.r = np.array([[object_radius]] * num_objects) if np.isscalar(object_radius) else object_radius
        self.m = np.array([[object_mass]] * num_objects) if np.isscalar(object_mass) else object_mass
        self.hw = coord_limits
        self.internal_steps = granularity
        self.eps = 1 / granularity
        self.res = size
        self.t = t

        self.fric_coeff = friction_coefficient
        self.v_rotation_angle = 2 * np.pi * 0.05

        if use_colors is None:
            if num_objects < 3:
                self.use_colors = False
            else:
                self.use_colors = True
        else:
            self.use_colors = use_colors

        if sprites:
            self.renderer = spriteworld_renderers.PILRenderer(
                image_size=(self.res, self.res),
                anti_aliasing=10,
            )

            shapes = ['triangle', 'square', 'circle', 'star_4']

            if not np.isscalar(object_radius):
                print("Scale elements according to radius of first element.")

            # empirical scaling rule, works for r = 1.2 and 2
            self.scale = self.r[0] / self.hw / 0.6
            self.shapes = np.random.choice(shapes, 3)
            self.draw_image = self.draw_sprites

        else:
            self.draw_image = self.draw_balls

        self.reset()

    def init_v(self, init_v_factor=None):
        """Randomly initialise velocities."""
        v = np.random.normal(size=(self.n, 2))
        v = v / np.sqrt((v ** 2).sum()) * .5

        if init_v_factor is not None:
            v = v * np.random.uniform(1/init_v_factor, init_v_factor)

        return v

    def init_x(self):
        """Initialize object positions without overlap and in bounds."""
        good_config = False
        while not good_config:
            x = np.random.rand(self.n, 2) * self.hw / 2 + self.hw / 4
            good_config = True
            for i in range(self.n):
                for z in range(2):
                    if x[i][z] - self.r[i] < 0:
                        good_config = False
                    if x[i][z] + self.r[i] > self.hw:
                        good_config = False

            for i in range(self.n):
                for j in range(i):
                    if norm(x[i] - x[j]) < self.r[i] + self.r[j]:
                        good_config = False
        return x

    def simulate_physics(self, actions):
        """Calculates physics for a single time step.

        What "physics" means is defined by the respective derived classes.

        Args:
            action (np.Array(float)): A 2D-float giving an x,y force to
                enact upon the first object.

        Returns:
            d_vs (np.Array(float)): Velocity updates for the simulation.

        """
        raise NotImplementedError

    def step(self, action=None, mass_center_obs=False):
        """Full step for the environment."""
        if action is not None:
            # Actions are implemented as hardly affecting the first object's v.
            self.v[0] = action * self.t
            actions = True

        else:
            actions = False

        for _ in range(self.internal_steps):
            self.x += self.t * self.eps * self.v

            if mass_center_obs:
                # Do simulation in center of mass system.
                c_body = np.sum(self.m * self.x, 0) / np.sum(self.m)
                self.x += self.hw / 2 - c_body

            self.v -= self.fric_coeff * self.m * self.v * self.t * self.eps
            self.v = self.simulate_physics(actions)

        img = self.draw_image()
        state = np.concatenate([self.x, self.v], axis=1)
        done = False

        return img, done, {"state:": state}

    def get_obs_shape(self):
        """Return image dimensions."""
        return (self.res, self.res, 3)

    def get_state_shape(self):
        """Get shape of state array."""
        state = np.concatenate([self.x, self.v], axis=1)
        return state.shape

    @staticmethod
    def ar(x, y, z):
        """Offset array function."""
        return z / 2 + np.arange(x, y, z, dtype='float')

    def draw_balls(self):
        """Render balls on canvas."""
        if self.n > 6:
            raise ValueError(
                'Max self.n implemented currently is 6.')

        img = np.zeros((self.res, self.res, 3), dtype='float')
        [I, J] = np.meshgrid(self.ar(0, 1, 1. / self.res) * self.hw,
                             self.ar(0, 1, 1. / self.res) * self.hw)

        colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, 1, 0], [1, 0, 1], [0, 1, 1]])

        for i in range(self.n):
            factor = np.exp(- (((I - self.x[i, 0]) ** 2 +
                                (J - self.x[i, 1]) ** 2) /
                               (self.r[i] ** 2)) ** 4)

            if self.use_colors:
                img[:, :, 0] += colors[i, 0] * factor
                img[:, :, 1] += colors[i, 1] * factor
                img[:, :, 2] += colors[i, 2] * factor

            else:
                idx = i % 3
                img[:, :, idx] += factor

        img[img > 1] = 1

        return img * 255.

    def draw_sprites(self):
        """Render sprites on the current locations."""

        s1 = Sprite(self.x[0, 0] / self.hw, 1 - self.x[0, 1] / self.hw,
                    self.shapes[0],
                    c0=255, c1=0, c2=0, scale=self.scale)
        s2 = Sprite(self.x[1, 0] / self.hw, 1 - self.x[1, 1] / self.hw,
                    self.shapes[1],
                    c0=0, c1=255, c2=0, scale=self.scale)
        s3 = Sprite(self.x[2, 0] / self.hw, 1 - self.x[2, 1] / self.hw,
                    self.shapes[2],
                    c0=0, c1=0, c2=255, scale=self.scale)

        sprites = [s1, s2, s3]
        img = self.renderer.render(sprites)

        return img

    def reset(self, init_v_factor=None):
        """Resets the environment to a new configuration."""
        self.v = self.init_v(init_v_factor)
        self.a = np.zeros_like(self.v)
        self.x = self.init_x()
        return self.draw_image()


class BillardsEnv(PhysicsEnv):
    """Billiards or Bouncing Balls environment."""

    def __init__(self, num_objects=3, object_radius=1., object_mass=1., coord_limits=10, granularity=5, size=32, t=1.,
                 init_v_factor=None, friction_coefficient=0., seed=None,
                 sprites=False, use_colors=None, drift=False):
        """Initialise arguments of parent class."""
        super().__init__(num_objects, object_radius, object_mass, coord_limits, granularity, size, t, init_v_factor,
                         friction_coefficient, seed, sprites, use_colors)

        # collisions is updated in step to measure the collisions of the balls
        self.collisions = 0

        # no collisions between objects!
        self.drift = drift

    def simulate_physics(self, actions):
        # F = ma = m dv/dt ---> dv = a * dt = F/m * dt
        v = self.v.copy()

        # check for collisions with wall
        for i in range(self.n):
            for z in range(2):
                next_pos = self.x[i, z] + (v[i, z] * self.eps * self.t)
                # collision at 0 wall
                if next_pos < self.r[i]:
                    self.x[i, z] = self.r[i]
                    v[i, z] = - v[i, z]
                # collision at hw wall
                elif next_pos > (self.hw - self.r[i]):
                    self.x[i, z] = self.hw - self.r[i]
                    v[i, z] = - v[i, z]

        if self.drift:
            return v

        # check for collisions with objects
        for i in range(self.n):
            for j in range(i):

                dist = norm((self.x[i] + v[i] * self.t * self.eps)
                            - (self.x[j] + v[j] * self.t * self.eps))

                if dist < (self.r[i] + self.r[j]):
                    if actions and j == 0:
                        self.collisions = 1

                    w = self.x[i] - self.x[j]
                    w = w / norm(w)

                    v_i = np.dot(w.transpose(), v[i])
                    v_j = np.dot(w.transpose(), v[j])

                    if actions and j == 0:
                        v_j = 0

                    new_v_i, new_v_j = self.new_speeds(
                        self.m[i], self.m[j], v_i, v_j)

                    v[i] += w * (new_v_i - v_i)
                    v[j] += w * (new_v_j - v_j)

                    if actions and j == 0:
                        v[j] = 0

        return v

    def new_speeds(self, m1, m2, v1, v2):
        """Implement elastic collision between two objects."""
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)
        return new_v1, new_v2

    def step(self, action=None):
        """Overwrite step functino to ensure collisions are zeroed beforehand."""
        self.collisions = 0
        img, done, info = super().step(action)
        return img, done, info
    
class BilliardDiscreteActionSpace():
    def __init__(self, n, max, min):
        self.n = n
        self.max = max
        self.min = min

class Task():
    """Defines a task for interactive environments.

    For all tasks defined here, actions correspond to direct movements of the
    controlled balls. Rewards are defined by the derived classes.
    """

    angular = 1. / np.sqrt(2)
    action_selection = [
        np.array([0., 0.]),
        np.array([1., 0.]),
        np.array([0., 1.]),
        np.array([angular, angular]),
        np.array([-1., 0.]),
        np.array([0., -1.]),
        np.array([-angular, -angular]),
        np.array([-angular, angular]),
        np.array([angular, -angular])]

    def __init__(self, env, num_stacked=4, greyscale=False, action_force=.3):
        """Initialise task.

        Args:
            env (Environment): Tasks have environments as attribute.
            num_stacked (int): Create a frame buffer of num_stacked images.
            greyscale (bool): Convert rgb images to 'greyscale'.
            action_force (float): Distance moved per applied action.
        """
        self.env = env

        # make controlled ball quasi-static
        self.env.m[0] = 10000

        if greyscale:
            self.frame_buffer = np.zeros(
                (*env.get_obs_shape()[:2], num_stacked))
            self.conversion = lambda x: np.sum(
                x * [[[0.3, 0.59, 0.11]]], 2, keepdims=True)
        else:
            sh = env.get_obs_shape()
            self.frame_buffer = np.zeros((*sh[:2], sh[2] * num_stacked))
            self.conversion = lambda x: x

        self.frame_channels = 3 if not greyscale else 1
        self.action_force = action_force

        self.action_space = BilliardDiscreteActionSpace(n=len(self.action_selection), max=len(self.action_selection), min=0)
        self.env.action_space = self.action_space

    def get_action_space(self):
        """Return number of available actions."""
        return len(self.action_selection)

    def get_framebuffer_shape(self):
        """Return shape of frame buffer."""
        return self.frame_buffer.shape

    def calculate_reward(self, state, action, env):
        """Abstract method. To be overwritten by derived classes."""
        raise NotImplementedError

    def resolve_action(self, _action, env=None):
        """Implement the effects of an action. Change this to change action."""
        action = self.action_selection[_action]
        action = action * self.action_force
        return action

    def step(self, _action):
        """Propagate env to next step."""
        action = self.resolve_action(_action)
        img, done, info = self.env.step(action)
        r = self.calculate_reward()
        return img, r, done, info

    def step_frame_buffer(self, _action=None):
        """Step environment with frame buffer."""
        action = self.resolve_action(_action)
        img, done, info = self.env.step(action)
        r = self.calculate_reward()

        img = self.conversion(img)
        c = self.frame_channels
        self.frame_buffer[:, :, :-c] = self.frame_buffer[:, :, c:]
        self.frame_buffer[:, :, -c:] = img

        return self.frame_buffer, r, done, info
    
    def reset(self):
        return self.env.reset()
    

class AvoidanceTask(Task):
    """Derived Task: Avoidance Task."""

    def calculate_reward(self,):
        """Negative sparse reward of -1 is given in case of collisions."""
        return -self.env.collisions

def make_billiards(size=64, num_objects=3, object_radius=1, object_mass=1., coord_limits=20, granularity=50, t=1., friction_coefficient=0.):
    base_env = BillardsEnv(size=size, num_objects=num_objects, object_radius=object_radius, object_mass=object_mass, 
                           coord_limits=coord_limits, granularity=granularity, t=t, friction_coefficient=friction_coefficient)
    env = AvoidanceTask(base_env, num_stacked=4, greyscale=False, action_force=0.6)

    return env