import pandas as pd

data = [
    (1, 189, 1),
    (2, 105, 5),
    (3, 65, 5),
    (4, 45, 3),
    (5, 29, -5),
    (6, 22, 0),
    (7, 18, -2),
    (8, 20, -2),
    (9, 18, 2),
    (10, 17, 5),
    (11, 18, 2),
    (12, 17, 1),
    (13, 19, 1),
    (14, 15, -1),
    (15, 21, -1),
    (16, 12, 2),
    (17, 12, 2),
    (18, 12, 0),
    (19, 12, 0),
    (20, 6, -2),
    (21, 7, -1),
    (22, 6, 0),
]

df = pd.DataFrame(data, columns=['candle_width', 'num_trades', 'profit'])

import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set_theme(style="whitegrid")

# Plot profit vs candle width
plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='candle_width', y='profit', marker='o', label='Profit')

# Plot number of trades on a secondary axis
ax2 = plt.gca().twinx()
sns.lineplot(data=df, x='candle_width', y='num_trades', marker='s', color='red', ax=ax2, label='Number of Trades')

# Labels and title
plt.title("Profit and Number of Trades vs Candle Width")
plt.xlabel("Candle Width")
plt.ylabel("Profit")
ax2.set_ylabel("Number of Trades")

# Show legends
plt.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig("plot.png", dpi=300)


