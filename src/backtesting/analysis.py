import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json

# take results_df

# PRINT

# returns over time plotted
# number of wins
# number of losses
# sharpe ratio
# alpha


# picture.to_png('/src/data/{env_variables}/graph1')
# variables.to_json('/src/data/{env_variables}/variables')

# merge results_df to full backtesting data dataframe
# write function that adds column that indicates trade outcome
# add fee column (fixed everytime we trade)

# summarisation:
# number of wins, number of losses

# baso same thing but plotted over time
#


def trial_analysis(full_backtesting_df, results_df, config, stats_dict, trial):

    trades_df = full_backtesting_df.join(results_df, how="left")
    outcome_df = calculate_realised_profit(
        trades_df, config["risk_to_reward_ratio"], config
    )
    stats_dict = generate_analysis_report(outcome_df, f"analysis/{trial}", full_backtesting_df, config, stats_dict, trial)

    return stats_dict

def calculate_realised_profit(df, risk_to_reward, config):
    """
    Adds two columns to the DataFrame:
      - 'realised_profit': The profit/loss for a trade based on future price data.
      - 'exit_index': The datetime index where the trade's take profit or stop loss was hit.
    Assumptions:
      - The DataFrame index is datetime.
      - 'trade': 0 = no trade, 1 = short, 2 = long.
      - Trade entry is assumed at the 'close' price on the signal row.
      - For long trades, the assumed intrabar price sequence is: open -> high -> low.
      - For short trades, the assumed sequence is: open -> low -> high.
    """
    n = len(df)

    # Convert necessary columns to NumPy arrays for speed.
    trade = df["trade"].values
    stop = df["stop_loss"].values
    tp = df["take_profit"].values
    close_p = df["Close"].values
    open_p = df["Open"].values
    high_p = df["High"].values
    low_p = df["Low"].values
    index_arr = df.index.values  # This should be a numpy array of np.datetime64[ns]

    # Create arrays for the output.
    realised_profit = np.full(n, np.nan, dtype=float)
    ambiguous_result = np.full(n, np.nan, dtype=int)

    # Specify the datetime dtype explicitly to avoid type-casting issues.
    exit_index = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")

    # Loop through each row.
    for i in range(n):
        if trade[i] == 0:
            continue  # No trade on this row.

        entry_price = close_p[i]
        trade_stop = stop[i]
        trade_tp = tp[i]

        # Short trade: look for a downward exit.
        if trade[i] == 1:
            for j in range(i + 1, n):
                op = open_p[j]
                hi = high_p[j]
                lo = low_p[j]
                if op <= trade_tp:
                    realised_profit[i] = entry_price - trade_tp
                    exit_index[i] = index_arr[j]
                    break
                elif op >= trade_stop:
                    realised_profit[i] = entry_price - trade_stop
                    exit_index[i] = index_arr[j]
                    break
                elif (lo <= trade_tp) and (hi >= trade_stop):
                    exit_index[i] = index_arr[j]
                    ambiguous_result[i] = 1
                    break
                elif lo <= trade_tp:
                    realised_profit[i] = entry_price - trade_tp
                    exit_index[i] = index_arr[j]
                    break
                elif hi >= trade_stop:
                    realised_profit[i] = entry_price - trade_stop
                    exit_index[i] = index_arr[j]
                    break

        # Long trade: look for an upward exit.
        elif trade[i] == 2:
            for j in range(i + 1, n):
                op = open_p[j]
                hi = high_p[j]
                lo = low_p[j]
                # Check conditions in a defined order:
                if op >= trade_tp:
                    realised_profit[i] = trade_tp - entry_price
                    exit_index[i] = index_arr[j]
                    break
                elif op <= trade_stop:
                    realised_profit[i] = trade_stop - entry_price
                    exit_index[i] = index_arr[j]
                    break
                elif (hi >= trade_tp) and (lo <= trade_stop):
                    ambiguous_result[i] = 1
                    exit_index[i] = index_arr[j]
                    break
                elif hi >= trade_tp:
                    realised_profit[i] = trade_tp - entry_price
                    exit_index[i] = index_arr[j]
                    break
                elif lo <= trade_stop:
                    realised_profit[i] = trade_stop - entry_price
                    exit_index[i] = index_arr[j]
                    break

    # Add the computed arrays as new columns.
    df["realised_profit"] = realised_profit
    df["win"] = np.where(
        df["realised_profit"] > 0,
        risk_to_reward,
        np.where(df["realised_profit"] < 0, -1, np.nan),
    )
    df["ambiguous_outcome"] = ambiguous_result
    df["exit_time"] = exit_index

    df["fee"] = np.where(
        df["win"] == np.nan,
        0,
        df["Close"] * (7 / 100000)
        * (
            (config["account_size"] * (config["risk_per_trade_percent"]) / 100)
            / df["ATR"]
        ),
    )
    df["fee_percent"] = 100 * df["fee"] / config["account_size"]

    ###
    ### Write outcome df to blob, concat all of the config as extra columns, also append code version column, unique runid

    return df


def compute_trade_statistics(df, risk_to_reward, config):
    """
    Compute statistics on trades.
    Only rows with a non-NaN 'realised_profit' are considered trades.
    Returns a dictionary of metrics and a trades DataFrame sorted by trade exit time.
    """
    # Filter for rows where a trade was executed (i.e. realised_profit is not NaN)
    trades_df = df.dropna(subset=["realised_profit"]).copy()
    total_trades = len(trades_df)

    # Wins/losses based on profit
    wins = trades_df[trades_df["realised_profit"] > 0]
    losses = trades_df[trades_df["realised_profit"] < 0]
    num_wins = len(wins)
    num_losses = len(losses)
    win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else np.nan

    total_profit = trades_df["realised_profit"].sum()
    avg_profit = trades_df["realised_profit"].mean() if total_trades > 0 else np.nan
    avg_win = wins["realised_profit"].mean() if num_wins > 0 else np.nan
    avg_loss = losses["realised_profit"].mean() if num_losses > 0 else np.nan

    # Profit factor: Sum of wins divided by absolute sum of losses.
    profit_factor = (
        wins["realised_profit"].sum() / -losses["realised_profit"].sum()
        if num_losses > 0
        else np.nan
    )

    # Cumulative Profit:
    cum_ret = np.ones(len(trades_df))
    if trades_df["realised_profit"][0] > 0:
        cum_ret[0] = (100 + risk_to_reward) / 100
    elif trades_df["realised_profit"][0] < 0:
        cum_ret[0] = 0.99
    for i in range(1, len(trades_df)):
        if trades_df["realised_profit"][i] > 0:
            cum_ret[i] = cum_ret[i-1] * (100 + risk_to_reward) / 100
        elif trades_df["realised_profit"][i] < 0:
            cum_ret[i] = cum_ret[i-1] * 0.99
        else:
            cum_ret[i] = cum_ret[i-1]
    cum_prof = cum_ret[-1] - 1

    # Cumulative Fees:
    cum_fees = cum_ret * trades_df['fee_percent']


    # Sharpe ratio for trades.
    std = (trades_df['win'] / 100).std()
    sharpe_ratio = (
        (((1 + (cum_prof - cum_fees.sum()/100)) ** (365 / (config["backtest_end_date_time"] - config["backtest_start_date_time"]).days) - 1) - (config["risk_free_annual_return"]/100)) /
        std
    )


    # Build the equity curve (cumulative sum of trade profits).
    trades_df = trades_df.sort_index()  # Assuming the index represents the exit time.
    equity = trades_df["win"].cumsum()

    # Maximum drawdown calculation.
    peak = equity.expanding().max()
    drawdown = equity - peak
    max_drawdown = drawdown.min()  # most negative drop (in absolute terms)
    # As a percentage:
    max_drawdown_pct = (drawdown / peak).min() if (peak > 0).all() else np.nan

    total_fees = trades_df["fee_percent"].sum()
    average_fee = trades_df[trades_df["fee"] != np.nan]["fee_percent"].mean()

    stats = {
        "Total Trades": total_trades,
        "Winning Trades": num_wins,
        "Losing Trades": num_losses,
        "Win Rate (%)": win_rate,
        "Total Fees": total_fees,
        "Average Fees": average_fee,
        "Profit % (Accounting for R2R)": num_wins * risk_to_reward - num_losses,
        "Profit % (Accounting for R2R and fees)": round(
            num_wins * risk_to_reward - num_losses - total_fees, 2
        ),
        "Cumulative Profit % (Accounting for R2R)": round(cum_prof * 100, 2),
        "Cumulative Profit % (Accounting for R2R and fees)": round(cum_prof * 100 - cum_fees.sum(), 2),
        "--- ONLY APPLICABLE IS RISK TO REWARD IS 1 ---": "",
        "Maximum Drawdown": max_drawdown,
        "Maximum Drawdown (%)": (
            max_drawdown_pct * 100 if not np.isnan(max_drawdown_pct) else np.nan
        ),
        "--- BELOW NOT CONFIGURED TO BE ACCURATE FOR OUR STRATEGY ---": "",
        f"Sharpe Ratio (Assuming annual 3% risk-free returns)": round(sharpe_ratio, 2) if std > 0 else np.nan,
        "Total Profit": total_profit,
        "Average Profit per Trade": avg_profit,
        "Average Win": avg_win,
        "Average Loss": avg_loss,
        "Profit Factor": profit_factor,
    }
    return stats, trades_df


# -------------------------------
# Plotting Functions
# -------------------------------
def plot_equity_curve(trades_df, output_folder):
    """
    Plots the equity curve (cumulative profit) over time.
    """
    trades_sorted = trades_df.sort_index()
    equity = trades_sorted["win"].cumsum()

    plt.figure(figsize=(10, 6))
    plt.plot(trades_sorted.index, equity, marker="o", linestyle="-")
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit")
    plt.grid(True)

    equity_curve_path = os.path.join(output_folder, "equity_curve.png")
    plt.savefig(equity_curve_path)
    plt.close()
    return equity_curve_path


# def plot_profit_histogram(trades_df, output_folder):
#     """
#     Plots a histogram (with kernel density) of trade profit/loss.
#     """
#     plt.figure(figsize=(10, 6))
#     sns.histplot(trades_df["realised_profit"], bins=30, kde=True)
#     plt.title("Histogram of Trade Profit/Loss")
#     plt.xlabel("Profit/Loss")
#     plt.ylabel("Frequency")

#     hist_path = os.path.join(output_folder, "profit_histogram.png")
#     plt.savefig(hist_path)
#     plt.close()
#     return hist_path


# TODO: Check logic here correct
def plot_trade_duration(df, output_folder):
    """
    (Optional) If your dataframe doesn't store each trade's duration, this example
    computes the time between consecutive trade exit times as a proxy for trade frequency.
    """
    trades_df = df.dropna(subset=["realised_profit"]).sort_index()
    # Calculate time difference between consecutive trades (in hours)
    durations = trades_df.index.to_series().diff().dropna().dt.total_seconds() / 3600
    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=30, kde=True)
    plt.title("Histogram of Time Between Trades (hours)")
    plt.xlabel("Time between trades (hours)")
    plt.ylabel("Frequency")

    duration_path = os.path.join(output_folder, "trade_duration_histogram.png")
    plt.savefig(duration_path)
    plt.close()
    return duration_path


def plot_equity_and_stock_price(trades_df, full_backtesting_df, output_folder):
    """
    Plots the equity curve (cumulative profit) alongside the stock price over time.
    Only considers the period where the equity curve is defined.
    """
    trades_sorted = trades_df.sort_index()
    equity = trades_sorted["win"].cumsum()

    # Define the date range based on trades_df
    start_date = trades_sorted.index.min()
    end_date = trades_sorted.index.max()

    # Ensure full_backtesting_df index is datetime and filter to matching range
    full_backtesting_df.index = pd.to_datetime(full_backtesting_df.index)
    filtered_stock_prices = full_backtesting_df.loc[start_date:end_date]

    plt.figure(figsize=(12, 6))
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot stock price within the filtered range
    ax1.plot(
        filtered_stock_prices.index,
        filtered_stock_prices["Close"],
        color="blue",
        alpha=0.7,
        label="Stock Price",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Stock Price", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    # Create secondary axis for equity curve
    ax2 = ax1.twinx()
    ax2.plot(
        trades_sorted.index,
        equity,
        color="green",
        marker="o",
        linestyle="-",
        label="Equity Curve",
    )
    ax2.set_ylabel("Cumulative Profit", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    # Add legends
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.title("Equity Curve vs Stock Price")

    # Save the plot
    output_path = os.path.join(output_folder, "equity_vs_stock_price.png")
    plt.savefig(output_path)
    plt.close()

    return output_path


def plot_stock_price_with_buy_sell(full_backtesting_df, trades_df, output_folder):
    """
    Plots the stock price from the first trade date onward with indications of where buys and sells occurred.
    """
    # Filter out rows where trade is 0 (no action)
    trades_filtered = trades_df[trades_df["trade"] != 0]

    # Get the first date when a trade occurs
    first_trade_date = trades_filtered.index[0]

    # Filter the full stock data to start from the first trade date
    full_backtesting_filtered = full_backtesting_df[
        full_backtesting_df.index >= first_trade_date
    ]

    # Get buy and sell points based on the 'trade' column
    buy_points = trades_filtered[trades_filtered["trade"] == 2]  # Buy actions
    sell_points = trades_filtered[trades_filtered["trade"] == 1]  # Sell actions

    plt.figure(figsize=(12, 6))

    # Plot the stock price from the first trade date onward
    plt.plot(
        full_backtesting_filtered.index,
        full_backtesting_filtered["Close"],
        color="blue",
        alpha=0.7,
        label="Stock Price",
    )

    # Plot buy points
    plt.scatter(
        buy_points.index,
        full_backtesting_df.loc[buy_points.index, "Close"],
        color="green",
        marker="^",
        label="Buy",
        s=100,
    )

    # Plot sell points
    plt.scatter(
        sell_points.index,
        full_backtesting_df.loc[sell_points.index, "Close"],
        color="red",
        marker="v",
        label="Sell",
        s=100,
    )

    # Add labels and legend
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price with Buy and Sell Indications")
    plt.legend(loc="best")

    # Save the plot
    output_path = os.path.join(output_folder, "stock_price_with_buy_sell.png")
    plt.savefig(output_path)
    plt.close()

    return output_path


# -------------------------------
# Report Generation
# -------------------------------
def generate_analysis_report(df, output_folder, full_backtesting_df, config, stats_dict, trial):
    """
    Computes statistics, generates plots, and writes a text report summarizing the analysis.
    """
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute trade statistics and filter trades.
    stats, trades_df = compute_trade_statistics(df, config["risk_to_reward_ratio"], config)

    # Generate plots.
    equity_curve_path = plot_equity_curve(trades_df, output_folder)
    duration_path = plot_trade_duration(df, output_folder)
    # equity_stock_price_path = plot_equity_and_stock_price(trades_df, full_backtesting_df, output_folder)
    stock_price_with_buy_sell_path = plot_stock_price_with_buy_sell(
        full_backtesting_df, trades_df, output_folder
    )

    # Build a text-based report.
    report_lines = []
    report_lines.append("Trade Analysis Report")
    report_lines.append("=" * 40)
    for key, value in stats.items():
        report_lines.append(f"{key}: {value}")
    report_lines.append("\nGraphs:")
    report_lines.append(f"Equity Curve: {equity_curve_path}")
    report_lines.append(f"Trade Duration Histogram: {duration_path}")
    # report_lines.append(f"Equity vs Stock Price: {equity_stock_price_path}")
    report_lines.append(
        f"Stock Price with Buy and Sell Indications: {stock_price_with_buy_sell_path}"
    )

    report_text = "\n".join(report_lines)

    # Write the report to a text file.
    report_file = os.path.join(output_folder, "trade_analysis_report.txt")
    with open(report_file, "w") as f:
        f.write(report_text)

    print("Analysis report saved to:", report_file)

    config_path = os.path.join(output_folder, "config.json")
    stats_path = os.path.join(output_folder, "stats.json")

    config["backtest_start_date_time"] = str(config["backtest_start_date_time"])
    config["backtest_end_date_time"] = str(config["backtest_end_date_time"])

    with open(config_path, "w") as json_file:
        json.dump(config, json_file, indent=4)

    with open(stats_path, "w") as json_file:
        json.dump(stats, json_file, indent=4)

    stats_dict[trial] = {**stats, **config}

    print(f"Config saved to: {config_path}")

    return stats_dict


def analysis(stats_dict):
    print('full analysis module - add analysis that crosses multiple trials')

    # Convert dictionary to DataFrame
    stats_df = pd.DataFrame.from_dict(stats_dict, orient='index')

    # Add trial names as a column
    stats_df.insert(0, "Trial", stats_df.index)

    # Reset index for proper DataFrame format
    stats_df.reset_index(drop=True, inplace=True)


    for r2r in stats_df['risk_to_reward_ratio'].unique():

        filtered_df = stats_df[stats_df['risk_to_reward_ratio'] == r2r]
        # add plotting syntax of CW vs profit after fees

        # save each plot to analysis/summary
