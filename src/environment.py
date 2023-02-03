# try out the wrappers
from acme import wrappers
from dm_control import suite
import numpy as np
import tree
from dm_env import specs


# environment wrappers
class NormilizeActionSpecWrapper(wrappers.EnvironmentWrapper):
    """Turn each dimension of the actions into the range of [-1, 1]."""

    def __init__(self, environment):
        super().__init__(environment)

        action_spec = environment.action_spec()
        self._scale = action_spec.maximum - action_spec.minimum
        self._offset = action_spec.minimum

        minimum = action_spec.minimum * 0 - 1.
        maximum = action_spec.minimum * 0 + 1.
        self._action_spec = specs.BoundedArray(
            action_spec.shape,
            action_spec.dtype,
            minimum,
            maximum,
            name=action_spec.name)

    def _from_normal_actions(self, actions):
        actions = 0.5 * (actions + 1.0)  # a_t is now in the range [0, 1]
        # scale range to [minimum, maximum]
        return actions * self._scale + self._offset

    def step(self, action):
        action = self._from_normal_actions(action)
        return self._environment.step(action)

    def action_spec(self):
        return self._action_spec


class MujocoActionNormalizer(wrappers.EnvironmentWrapper):
    """Rescale actions to [-1, 1] range for mujoco physics engine.
    For control environments whose actions have bounded range in [-1, 1], this
      adaptor rescale actions to the desired range. This allows actor network to
      output unscaled actions for better gradient dynamics.
    """

    def __init__(self, environment, rescale='clip'):
        super().__init__(environment)
        self._rescale = rescale

    def step(self, action):
        """Rescale actions to [-1, 1] range before stepping wrapped environment."""
        if self._rescale == 'tanh':
            scaled_actions = tree.map_structure(np.tanh, action)
        elif self._rescale == 'clip':
            scaled_actions = tree.map_structure(lambda a: np.clip(a, -1., 1.), action)
        else:
            raise ValueError('Unrecognized scaling option: %s' % self._rescale)
        return self._environment.step(scaled_actions)


def get_flat_obs(time_step):
    # flatten all of the observations into a single array
    flat_obs = tree.flatten(time_step.observation)
    # combine all of the observations into a single array
    obs = np.concatenate(flat_obs)
    return obs