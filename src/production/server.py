from fastapi import FastAPI
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import datetime as dt
import uvicorn as uv

from production.remote import process_api_data

app = FastAPI()

@app.get('/')
def read_root():
    body = "Working!"
    return {'message': body}


@app.post('/train_and_predict')
def train_and_predict(json: dict):
    live_df = process_api_data(json)
    
    return live_df.to_dict()


