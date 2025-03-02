from fastapi import FastAPI, HTTPException, Request
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime
import datetime as dt
import uvicorn as uv
import os
import json
from pydantic import BaseModel
from typing import Any
import sys
import traceback

from production.remote.process_api_data import process_api_data
from production.make_features import make_features
from production.models.classifier import classifier
from production.models.ATR import ATR_prod_refactor

import logging

# Create logs directory
os.makedirs("/app/logs", exist_ok=True)

# Configure file logging for complete, untruncated logs
file_handler = logging.FileHandler("/app/logs/full_debug.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

# Configure console logging
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

# Set up root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Get a specific logger for your app
app_logger = logging.getLogger(__name__)

app = FastAPI()

CONFIG_PATH = "/app/production/config_production.json"

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

# Define Pydantic model for the incoming request
class PredictionRequest(BaseModel):
    data: str

# Create a separate endpoint that will just log the request without validation
@app.post('/log_raw_request')
async def log_raw_request(request: Request):
    """
    Endpoint that logs the raw request body without any validation
    """
    try:
        # Get the raw request body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Save to a file with timestamp to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"/app/logs/raw_request_{timestamp}.json"
        with open(file_path, "w") as f:
            f.write(body_str)
        
        # Log the request body in full
        app_logger.info(f"FULL REQUEST BODY START >>>>>")
        app_logger.info(body_str)
        app_logger.info(f"<<<<< FULL REQUEST BODY END")
        
        # Try to parse JSON
        try:
            json_data = json.loads(body_str)
            return {
                "status": "success",
                "message": f"Request logged to {file_path}",
                "has_data_field": "data" in json_data,
                "content_type": request.headers.get("content-type", "not specified")
            }
        except json.JSONDecodeError as e:
            return {
                "status": "invalid_json",
                "message": f"Request logged to {file_path}, but JSON is invalid: {str(e)}",
                "content_type": request.headers.get("content-type", "not specified")
            }
    except Exception as e:
        app_logger.exception(f"Error logging request: {str(e)}")
        return {"status": "error", "message": str(e)}

# Add middleware to capture all request bodies
@app.middleware("http")
async def log_all_requests(request: Request, call_next):
    """
    Middleware that logs all request bodies
    """
    request_path = request.url.path
    app_logger.info(f"Request received: {request.method} {request_path}")
    
    # For specific endpoints, log the complete request body
    if request_path in ['/train_and_predict']:
        try:
            # Need to read the body
            body = await request.body()
            
            # Convert to string for logging
            body_str = body.decode('utf-8')
            
            # Save to a dedicated file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"/app/logs/request_{timestamp}.json"
            with open(file_path, "w") as f:
                f.write(body_str)
            
            # Also log to console and log file
            app_logger.info(f"Request to {request_path} - FULL BODY SAVED TO {file_path}")
            
            # We need to set the body back so that it can be read by the endpoint
            # Create a new Request with the same body
            async def receive():
                return {"type": "http.request", "body": body}
            
            request._receive = receive
        except Exception as e:
            app_logger.exception(f"Error in request logging middleware: {str(e)}")
    
    # Continue with request processing
    response = await call_next(request)
    return response

# Basic root endpoint
@app.get('/')
def read_root():
    app_logger.info("Root endpoint accessed")
    return {'message': "API is running. Use /train_and_predict for predictions or /log_raw_request to debug request format."}

# Main prediction endpoint
@app.post('/train_and_predict')
async def train_and_predict(request: Request):
    """
    Main prediction endpoint that logs the raw request first, then tries to process it
    """
    app_logger.info("Processing prediction request")
    
    try:
        # Get raw request body
        body = await request.body()
        body_str = body.decode('utf-8')
        
        # Save raw request to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file_path = f"/app/logs/prediction_request_{timestamp}.json"
        with open(raw_file_path, "w") as f:
            f.write(body_str)
        
        app_logger.info(f"Prediction request saved to {raw_file_path}")
        app_logger.info("Raw request body: " + body_str)
        
        # Parse JSON
        try:
            json_data = json.loads(body_str)
            
            # Check for data field
            if "data" not in json_data:
                error_msg = "Request JSON is missing required 'data' field"
                app_logger.error(error_msg)
                return {"error": error_msg}
            
            # Get the data string
            data = json_data["data"]
            app_logger.info(f"Extracted data field (first 100 chars): {data[:min(100, len(data))]}...")
            
            # Get target_width from config
            target_width = config.get("target_width", 10)
            app_logger.info(f"Using target_width: {target_width}")
            
            # Process data
            app_logger.info("Starting process_api_data...")
            live_df = process_api_data(data)
            app_logger.info(f"API data processed, DataFrame shape: {live_df.shape if live_df is not None else 'None'}")
            
            app_logger.info("Starting make_features...")
            features_df = make_features(live_df, target_width)
            app_logger.info(f"Features created, DataFrame shape: {features_df.shape if features_df is not None else 'None'}")
            
            app_logger.info("Starting classifier prediction...")
            trade = classifier(features_df)
            app_logger.info(f"Classifier result: trade={trade}")
            
            if trade != 0:
                app_logger.info("Trade signal detected, calculating ATR...")
                stop_loss, take_profit, atr = ATR_prod_refactor(
                    features_df, trade, config["risk_to_reward_ratio"]
                )
                app_logger.info(f"ATR calculation complete: SL={stop_loss}, TP={take_profit}, ATR={atr}")
            else:
                app_logger.info("No trade signal (trade=0), skipping ATR calculation")
                stop_loss = None
                take_profit = None
                atr = None
            
            time = str(features_df.tail(1).index[0])
            app_logger.info(f"Final prediction results: {time}, trade={trade}, stop_loss={stop_loss}, take_profit={take_profit}, ATR={atr}")
            
            return {
                "time": time,
                "trade": trade,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "ATR": atr
            }
            
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON: {str(e)}"
            app_logger.error(error_msg)
            return {"error": error_msg}
            
    except Exception as e:
        app_logger.exception(f"Error in train_and_predict: {str(e)}")
        error_traceback = traceback.format_exc()
        app_logger.error(f"Full traceback: {error_traceback}")
        return {
            "error": "Error during prediction",
            "details": str(e),
            "traceback": error_traceback
        }