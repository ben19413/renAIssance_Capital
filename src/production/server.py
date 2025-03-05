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
import logging
app = FastAPI()

import sys
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any

# Configure logging
logger = logging.getLogger("my_app")  # Custom logger name
logger.setLevel(logging.DEBUG)  # Set logging level to DEBUG

# Create a handler that logs to stdout (Docker will capture this)
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)

# Define a log format
formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
stream_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(stream_handler)

# Prevent duplicate log entries by removing default handlers
logger.propagate = False

CONFIG_PATH = "/app/production/config_production.json"

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

@app.get('/')
def read_root():
    body = "Working!"
    return {'message': body}


@app.post('/train_and_predict')
async def train_and_predict(request: Request):
    logger.info("POST request received")
    body = await request.body()

    
    #logger.info("Raw body bytes: %s", body_bytes)

    # logger.info(body)

    body_str = body.decode('utf-8')
    body_str = body_str.lstrip('\x00')
    logger.info(f"Body length: {len(body_str)}")
    body_str = body_str.strip()

    logger.info(f"Body length after strip: {len(body_str)}")
    logger.info(f"Raw body content: {body_str}")
    logger.info("PREJSONPARSE")
    logger.info(body_str)
    logger.info(type(body_str))

    body_str = json.loads(body_str) 

    logger.info("POSTJSONPARSE")
    logger.info(body_str)
    logger.info(type(body_str))

    live_df = process_api_data(body_str)
    features_df = make_features(live_df, config["target_width"])
    features_df.index = pd.to_datetime(features_df["Time"])
    trade = classifier(features_df)

    prediction_df = pd.DataFrame(
                {
                    "trade": [trade]
                }
            )
    prediction_df.index = features_df.tail(1).index

    orders_df = ATR(features_df, prediction_df, config["risk_to_reward_ratio"])
    print(orders_df)
    print(orders_df.shape)

    time = str(orders_df.tail(1).index[0])
    stop_loss = float(orders_df.tail(1)["stop_loss"])
    take_profit = float(orders_df.tail(1)["take_profit"])
    atr = float(orders_df.tail(1)["ATR"])

    return {
        "time": time,
        "trade": trade,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "ATR": atr
    }