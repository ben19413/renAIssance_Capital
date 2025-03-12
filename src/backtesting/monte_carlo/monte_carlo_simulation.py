import numpy as np
import os
import matplotlib.pyplot as plt
np.random.seed(42) 


def percentage_runs_passed(win_rate_percentage, 
                           win_size_as_percentage,
                           loss_size_as_percentage,
                           profit_goal_percentage, 
                           loss_limit_percentage, 
                           num_runs=1000, 
                           cumulative=True):
    num_wins = 0
    num_trades = 0
    for run in range(num_runs):
        account_size = 1  
        while True:  
            winning_trade = ((win_rate_percentage / 100) > np.random.rand())
            num_trades += 1
            if cumulative:
                if winning_trade:
                    account_size *= 1 + win_size_as_percentage / 100 - (7 / 100000)
                else:
                    account_size *= 1 - loss_size_as_percentage / 100 - (7 / 100000)
            else:
                if winning_trade:
                    account_size += win_size_as_percentage / 100 - (7 / 100000)
                else:
                    account_size -= loss_size_as_percentage / 100 - (7 / 100000)

            if account_size >= 1 + profit_goal_percentage / 100:
                num_wins += 1 
                break
            if account_size <= 1 - loss_limit_percentage / 100:
                break  

    percentage_passed = (num_wins / num_runs) * 100
    avg_trades = num_trades / num_runs
    
    return percentage_passed, avg_trades

# Define parameter range
win_loss_values = np.arange(0.4, 5.2, 0.2)
percent_passed_list = []
avg_trades_list = []

# Run simulations for each value
for value in win_loss_values:
    percent_passed, avg_trades = percentage_runs_passed(
        win_rate_percentage=55, 
        win_size_as_percentage=value, 
        loss_size_as_percentage=value, 
        profit_goal_percentage=10, 
        loss_limit_percentage=10, 
        num_runs=100000
    )
    percent_passed_list.append(percent_passed)
    avg_trades_list.append(avg_trades)

# Create and save the plot with dual axes
os.makedirs("plots", exist_ok=True)  # Ensure the folder exists

fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot percentage of runs passed on the first axis
ax1.set_xlabel("Win/Loss Size as Percentage")
ax1.set_ylabel("Percentage of Runs Passed (%)", color="tab:blue")
ax1.plot(win_loss_values, percent_passed_list, label="Percentage of Runs Passed (%)", color="tab:blue", marker="o")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# Create second y-axis to plot the average number of trades
ax2 = ax1.twinx()
ax2.set_ylabel("Average Trades per Run", color="tab:green")
ax2.plot(win_loss_values, avg_trades_list, label="Average Trades per Run", color="tab:green", marker="s")
ax2.tick_params(axis="y", labelcolor="tab:green")

# Add grid and title
ax1.grid(True)
plt.title("Trading Simulation Results")

# Save the figure
fig.tight_layout()
plt.savefig("plots/trading_simulation_dual_axes.png")
plt.show()