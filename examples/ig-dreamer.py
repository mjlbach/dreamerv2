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
    'prefill': 1e5,
    'actor_ent': 3e-3,
    'loss_scales.kl': 1.0,
    'discount': 0.99,
    'task': 'igibson',
    'encoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb'},
    'decoder': { 'mlp_keys': '$^', 'cnn_keys': 'rgb'},
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
}).parse_flags()

env_config = os.path.join(igibson.root_path, "examples/configs/behavior_onboard_sensing_fetch.yaml")
env = BehaviorEnv(env_config)
# env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
@profile
def my_func():
    dv2.train(env, config)

my_func()
