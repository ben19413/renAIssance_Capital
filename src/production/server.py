from fastapi import FastAPI
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import datetime as dt
import uvicorn as uv
import os
import json

from production.remote.process_api_data import process_api_data
from production.make_features import make_features
from production.models.classifier import classifier
from production.models.ATR import ATR_prod_refactor

app = FastAPI()

CONFIG_PATH = "/app/production/config_production.json"

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

@app.get('/')
def read_root():
    body = "Working!"
    return {'message': body}


@app.post('/train_and_predict')
def train_and_predict(json: dict):

    live_df = process_api_data(json)
    features_df = make_features(live_df)

    trade = classifier(features_df)

    if trade != 0:
        stop_loss, take_profit, atr = ATR_prod_refactor(
            features_df, trade, config["risk_to_reward_ratio"]
        )
    else:
        stop_loss = None
        take_profit = None
        atr = None

    time = str(features_df.tail(1).index[0])

    return {
        "time": time,
        "trade": trade,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "ATR": atr
    }