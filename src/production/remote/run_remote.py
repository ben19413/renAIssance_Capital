# retrieve price data via:
# call process_api_data (input: data sent via api trigger) (output: data in same format as data after filtering in backtesting)

# make features
# call train classifier (input: data) (output: buy, sell, no trade)
# if not no trade:
#       call train ATR (input: data) (output: take profit, stop loss)
# else:
#       give stop loss and take profit arbitary values
# Return prediction, stop loss, take profit

import uvicorn as uv
from fastapi import FastAPI
import random


app = FastAPI()

@app.post('/train_and_predict')
def train_and_predict(data: dict):

    return {'test':'test'}





# Function to generate OHLC data
def generate_ohlc_data(num_observations=50):
    ohlc_data = {
        "open": [],
        "high": [],
        "low": [],
        "close": []
    }
    
    # Initialize a random starting price
    current_price = random.uniform(100, 200)
    
    for _ in range(num_observations):
        # Generate random OHLC values around the current price
        open_price = current_price
        high_price = open_price + random.uniform(0, 5)
        low_price = open_price - random.uniform(0, 5)
        close_price = random.uniform(low_price, high_price)
        
        # Append to dictionary
        ohlc_data["open"].append(round(open_price, 2))
        ohlc_data["high"].append(round(high_price, 2))
        ohlc_data["low"].append(round(low_price, 2))
        ohlc_data["close"].append(round(close_price, 2))
        
        # Update current price for next iteration
        current_price = close_price

    return ohlc_data




url = 'http://0.0.0.0:8000/train_and_predict/'




payload = json.dumps(generate_ohlc_data())
response = requests.post(url, data=payload)
print(response.json())
