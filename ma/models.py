# common library
import pandas as pd
import numpy as np
import time

# RL models from stable-baselines
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

import ma.config
from ma.env_stocktrading import StockTradingEnv

from stable_baselines3 import A2C
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3 import SAC
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        train_DDPG()
            the implementation for DDPG algorithm
        train_TD3()
            the implementation for TD3 algorithm
        train_SAC()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env):
        self.env = env

    # build model of particular type
    def get_model(self,model_name,policy="MlpPolicy",policy_kwargs=None,model_kwargs=None,verbose=1,):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)

        # build DRL model
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    @staticmethod
    def train_model(model, tb_log_name, total_timesteps):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        return model

    @staticmethod
    def DRL_prediction(model_name, model_file, test_env, test_obs):
        """make a prediction"""
        model = MODELS[model_name].load(model_file)
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(test_env.df.index.unique())):
            # generate action
            action, _states = model.predict(test_obs)
            print("action {}".format(action))
            # get reward from environment
            test_obs, rewards, dones, info = test_env.step(action)
            print(test_obs)

            if i == (len(test_env.df.index.unique())-2):
                account_memory = test_env.save_asset_memory()
                actions_memory = test_env.save_action_memory()
                print("Log test result")
            if dones:
                print("hit end!")
                break
        return account_memory, actions_memory