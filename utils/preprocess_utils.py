import gymnasium as gym
import numpy as np

from typing import Any, Callable
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, WrapperObsType

from typing import Optional, Union, Tuple



import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

class TransformObservation(
    gym.ObservationWrapper[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Applies a function to the ``observation`` received from the environment's :meth:`Env.reset` and :meth:`Env.step` that is passed back to the user.

    The function :attr:`func` will be applied to all observations.
    If the observations from :attr:`func` are outside the bounds of the ``env``'s observation space, provide an updated :attr:`observation_space`.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.TransformObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import TransformObservation
        >>> import numpy as np
        >>> np.random.seed(0)
        >>> env = gym.make("CartPole-v1")
        >>> env.reset(seed=42)
        (array([ 0.0273956 , -0.00611216,  0.03585979,  0.0197368 ], dtype=float32), {})
        >>> env = gym.make("CartPole-v1")
        >>> env = TransformObservation(env, lambda obs: obs + 0.1 * np.random.random(obs.shape), env.observation_space)
        >>> env.reset(seed=42)
        (array([0.08227695, 0.06540678, 0.09613613, 0.07422512]), {})

    Change logs:
     * v0.15.4 - Initially added
     * v1.0.0 - Add requirement of ``observation_space``
    """

    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        func: Callable[[ObsType], Any],
        # observation_space: gym.Space[WrapperObsType] | None,
        observation_space: Optional[gym.Space[WrapperObsType]] = None,
    ):
        """Constructor for the transform observation wrapper.

        Args:
            env: The environment to wrap
            func: A function that will transform an observation. If this transformed observation is outside the observation space of ``env.observation_space`` then provide an `observation_space`.
            observation_space: The observation spaces of the wrapper, if None, then it is assumed the same as ``env.observation_space``.
        """
        gym.utils.RecordConstructorArgs.__init__(
            self, func=func, observation_space=observation_space
        )
        gym.ObservationWrapper.__init__(self, env)

        if observation_space is not None:
            self.observation_space = observation_space

        self.func = func

    def observation(self, observation: ObsType) -> Any:
        """Apply function to the observation."""
        return self.func(observation)


class RepeatAction(gym.Wrapper):
    """
    Repeat the same action a certain number of times.
    Often, in a game, a couple of sequential frames do not differ that much from each other, so repeat the chosen action.
    """

    def __init__(self, env=None, repeat=4, clip_rewards=False, fire_first=False):
        """
        :param env:
        :param repeat:
        :param fire_first: If the rl_agent need to start the game by pressing "fire" button
        """
        # fire_first: in certain envs the rl_agent have to fire to start the env, as in pong
        # the rl_agent can figure it out alone sometimes
        super(RepeatAction, self).__init__(env)
        self.env = env
        self.repeat = repeat
        self.clip_rewards = clip_rewards
        self.shape = env.observation_space.low.shape
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, truncated, terminated, info = self.env.step(action)
            if self.clip_rewards:
                # clip the reward in -1, 1, then take first element (we need the scalar, not an array)
                reward = np.sign(reward)  # np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward

            done = truncated or terminated
            if done:
                break
        return obs, total_reward, truncated, terminated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.fire_first:
            # get_action_meanings returns a list of strings (['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE'])
            fire_act_idx = self.env.unwrapped.get_action_meanings().index("FIRE")
            obs, _, _, _ = self.env.step(fire_act_idx)
        return obs, info
    
class RescaleObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Normalize the observation from [0, 255] to be in the range [0, 1].

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.min()
        0
        >>> env.observation_space.max()
        255
        >>> normalize_env = RescaleObservation(env)
        >>> normalize_env.observation_space.min()
        0
        >>> normalize_env.observation_space.max()
        1


    Change logs:
     * v1.0.0 - Initially added
    """

    # shape: int | tuple[int, ...]):
    def __init__(self, env: gym.Env[ObsType, ActType], rescale_value=255.0):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        # assert np.prod(shape) == np.prod(env.observation_space.shape)

        # assert isinstance(shape, tuple)
        # assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        # assert all(x > 0 or x == -1 for x in shape)

        # new_observation_space = spaces.Box(
        #     low=np.zeros(shape, dtype=env.observation_space.dtype),
        #     high=np.ones(shape, dtype=env.observation_space.dtype),
        #     dtype=env.observation_space.dtype,
        # )
        # self.shape = shape

        new_observation_space = spaces.Box(
            low=env.observation_space.low / rescale_value,
            high=env.observation_space.high / rescale_value,
            shape=env.observation_space.shape,
            dtype=np.float32,
        )
        # gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        TransformObservation.__init__(
            self,
            env=env,
            func=lambda obs: obs / rescale_value,
            observation_space=new_observation_space,
        )

class ReshapeObservation(
    TransformObservation[WrapperObsType, ActType, ObsType],
    gym.utils.RecordConstructorArgs,
):
    """Reshapes Array based observations to a specified shape.

    A vector version of the wrapper exists :class:`gymnasium.wrappers.vector.RescaleObservation`.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.wrappers import ReshapeObservation
        >>> env = gym.make("CarRacing-v2")
        >>> env.observation_space.shape
        (96, 96, 3)
        >>> reshape_env = ReshapeObservation(env, (24, 4, 96, 1, 3))
        >>> reshape_env.observation_space.shape
        (24, 4, 96, 1, 3)

    Change logs:
     * v1.0.0 - Initially added
    """

    # shape: int | tuple[int, ...]):
    def __init__(
        self, env: gym.Env[ObsType, ActType], shape: Union[int, Tuple[int, ...]]
    ):
        """Constructor for env with ``Box`` observation space that has a shape product equal to the new shape product.

        Args:
            env: The environment to wrap
            shape: The reshaped observation space
        """
        assert isinstance(env.observation_space, spaces.Box)
        assert np.prod(shape) == np.prod(env.observation_space.shape)

        assert isinstance(shape, tuple)
        assert all(np.issubdtype(type(elem), np.integer) for elem in shape)
        assert all(x > 0 or x == -1 for x in shape)

        new_observation_space = spaces.Box(
            low=np.reshape(np.ravel(env.observation_space.low), shape),
            high=np.reshape(np.ravel(env.observation_space.high), shape),
            shape=shape,
            dtype=env.observation_space.dtype,
        )
        self.shape = shape
        gym.utils.RecordConstructorArgs.__init__(self, shape=shape)
        TransformObservation.__init__(
            self,
            env=env,
            # func=lambda obs: np.reshape(obs, shape),
            func=lambda obs: np.transpose(obs, (2,0,1)),
            observation_space=new_observation_space,
        )
