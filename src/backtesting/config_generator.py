import json
import os

def generate_config(
    backtest_start_date_time,
    backtest_end_date_time,
    account_size,
    risk_per_trade_percent,
    risk_free_annual_return,
    instrument,
    training_period_data_size,
    target_widths,
    risk_to_reward_ratios
):
    config = {
        "general": {
            "backtest_start_date_time": backtest_start_date_time,
            "backtest_end_date_time": backtest_end_date_time,
            "account_size": account_size,
            "risk_per_trade_percent": risk_per_trade_percent,
            "risk_free_annual_return": risk_free_annual_return,
        },
        "trials": {}
    }
    
    for tw in target_widths:
        for rr in risk_to_reward_ratios:
            key = f"RR{rr}, CW{tw}"
            config["trials"][key] = {
                "instrument": instrument,
                "training_period_data_size": training_period_data_size,
                "target_width": tw,
                "risk_to_reward_ratio": rr
            }
    
    return config

def save_config(config, filename="/app/src/config_development.json"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)





config = generate_config(
    backtest_start_date_time="2022-09-17 18:00",
    backtest_end_date_time="2025-02-01 18:00",
    account_size=1000,
    risk_per_trade_percent=1,
    risk_free_annual_return=3,
    instrument="EURUSD.csv",
    training_period_data_size=4545,
    target_widths=[3, 5, 8, 10, 12, 15, 20, 25, 30],
    risk_to_reward_ratios=[0.5, 1, 2, 3, 5, 7, 10, 12, 15, 17, 20]
)
save_config(config)