from fastapi import FastAPI, Request
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
from production.models.ATR import ATR

app = FastAPI()


CONFIG_PATH = "/app/production/config_production.json"

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

@app.get('/')
def read_root():
    body = "Working!"
    return {'message': body}


@app.post('/train_and_predict')
async def train_and_predict(request: Request):
    body = await request.body()
    body_str = body.decode('utf-8')
    body_str = body_str.lstrip('\x00')
    body_str = body_str.strip()
    body_str = json.loads(body_str) 
    live_df = process_api_data(body_str)
    live_df.index = pd.to_datetime(live_df["Time"])
    features_df = make_features(live_df, config["target_width"])
    trade = classifier(features_df)

    prediction_df = pd.DataFrame(
                {
                    "trade": [trade]
                }
            )
    prediction_df.index = features_df.tail(1).index

    orders_df = ATR(features_df, prediction_df, config["risk_to_reward_ratio"])

    time = str(orders_df.tail(1).index[0])
    stop_loss = float(orders_df.tail(1)["stop_loss"].iloc[0])
    take_profit = float(orders_df.tail(1)["take_profit"].iloc[0])
    atr = float(orders_df.tail(1)["ATR"].iloc[0])

    if trade == 0:
        stop_loss = 0
        take_profit = 0
        atr = 0

    return {
        "time": time,
        "trade": trade,
        "stop_loss": stop_loss if stop_loss is not np.nan else 0,
        "take_profit": take_profit if take_profit is not np.nan else 0,
        "ATR": atr if atr is not np.nan else 0
    }