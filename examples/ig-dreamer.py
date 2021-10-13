import gym
import os
import igibson
import dreamerv2.api as dv2
from igibson.envs.behavior_env import BehaviorEnv
from memory_profiler import profile

config = dv2.defaults.update({
    'logdir': '~/logdir/igibson-plan2xplore',
    'log_every': 1e3,
    'log_keys_video': ['rgb'],
    'train_every': 10,
    'prefill': 1e2,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'task': 'igibson',
    'encoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb', 'cnn_depth': 10},
    'decoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb', 'cnn_depth': 10},
    'action_repeat': 2,
    'eval_every': 1e4,
    'prefill': 100,
    'pretrain': 100,
    'clip_rewards': 'identity',
    'pred_discount': False,
    'replay.prioritize_ends': False,
    'grad_heads': ['decoder', 'reward'],
    'rssm': {'hidden': 200, 'deter': 200},
    'model_opt': {'lr' : 3e-4},
    'actor_opt': {'lr' : 8e-5},
    'critic_opt': {'lr' : 8e-5},
    'actor_ent': 1e-4,
    'kl': {'free' : 1.0},
    'replay': {'capacity': 100}
}).parse_flags()

env_config = "behavior_onboard_sensing.yaml"
env = BehaviorEnv(env_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
dv2.train(env, config)
