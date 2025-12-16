import gymnasium as gym
from cleanrl_utils.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from utils.preprocess_utils import *


def make_env(
    env,
    seed=0,
    rgb_array=True,
    rgb=True,
    stack=0,
    no_op=0,
    action_repeat=0,
    max_and_skip=False,
    episodic_life=False,
    clip_reward=False,
    check_fire=True,
    color_transform="standard",
    filter_dict=None,
    time_limit: int = 0,
    idx=0,
    capture_video=False,
    run_name="",
):
    def thunk(env=env):
        # print('Observation space: ', env.observation_space, 'Action space: ', env.action_space)
        # env = gym.make(env_id)
        # env = CarRacing(continuous=False, background='red')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        if no_op > 0:
            env = NoopResetEnv(env, noop_max=30)
        if action_repeat > 0:
            if max_and_skip:
                env = MaxAndSkipEnv(env, skip=action_repeat if action_repeat > 1 else 4)
            else:
                env = RepeatAction(env, repeat=action_repeat)
        if episodic_life:
            env = EpisodicLifeEnv(env)
        if check_fire and "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        if clip_reward:
            env = ClipRewardEnv(env)
        # if filter_dict:
        #     env = FilterFromDict(env, filter_dict)
        if rgb_array:
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            # if color_transform != "standard":
            #     env = ColorTransformObservation(env, color=color_transform)
            env = RescaleObservation(env, rescale_value=255.0)
            if rgb:
                #     env = PreprocessFrameRGB((84, 84, 3), env)  #
                # env = ReshapeObservation(env, (3, 96, 96)) # replace with env.observation_space.shape[1],
                
                # env = ReshapeObservation(env, (shape[2], shape[0], shape[1]))
                env = ReshapeObservation(env, (3, 84, 84))
            if not rgb:
                # env = gym.wrappers.ResizeObservation(env, (84, 84))
                env = gym.wrappers.GrayScaleObservation(env)
        # env = NormalizeFrames(env)
        # env = gym.wrappers.GrayScaleObservation(env)
        if stack > 1:
            env = gym.wrappers.FrameStack(env, stack)  # (4, 3, 84, 84)
        if time_limit > 0:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=time_limit)
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk





def make_env_standard(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        return env

    return thunk




def make_env_standard_cleanrl(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 1)
        return env

    return thunk


