from os import stat
import pandas as pd
import numpy as np

import ma.config
from ma.ssedataloader import SSEDataLoader
from ma.preprocessors import data_split, FeatureEngineer
from ma.env_stocktrading import StockTradingEnv
from ma.models import DRLAgent
from ma.backtest import backtest_stats


def train_agent():
    """
    train an agent
    """
    use_raw_data = False
    if use_raw_data == True:
        print("==============Load Raw Data===========")
        # data from Shanghai Stock Exchange
        df = SSEDataLoader().load_data()

        print("==============Feature Engineering===========")
        # calculate factors form raw data
        fe = FeatureEngineer(
            use_technical_indicator=False,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=False,
            user_defined_feature=False,
        )
        processed_df = fe.preprocess_data(df)
    else:
        print("==============Load processed Data===========")
        # data from Shanghai Stock Exchange
        processed_df = SSEDataLoader().load_data()


    print("==============Build Environment===========")
    # split dataset into train & test set
    train_df, test_df = data_split(processed_df, train_ratio=0.8)
    state_space = (3 + len(config.TECHNICAL_INDICATORS_LIST))
    print("state_space: {}".format(state_space))

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "reward_scaling": 1e-4,
        "state_space": state_space,
        "action_space": 1,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST
        }

    e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    e_trade_gym = StockTradingEnv(df=test_df, **env_kwargs)
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    print("Finish environment building")

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    agent = DRLAgent(env=env_train)
    # sac model
    model_type = "sac"
    model = agent.get_model(model_type)
    trained_model = DRLAgent.train_model(model=model, tb_log_name=model_type, total_timesteps=8000)
    model_file = "./{}/{}_{}".format(config.TRAINED_MODEL_DIR, model_type, now)
    trained_model.save(model_file)
    print("Finish model {} training".format(trained_model))

    print("==============Model Testing===========")
    # make prediction in test set 
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model_name=model_type, model_file=model_file, test_env=e_trade_gym, test_obs=obs_trade
    )
    df_account_value.to_csv("./{}/df_account_value_{}.csv".format(config.RESULTS_DIR, now))
    df_actions.to_csv("./{}/df_actions_{}.csv".format(config.RESULTS_DIR, now))
    print("Finish model {} testing".format(trained_model))