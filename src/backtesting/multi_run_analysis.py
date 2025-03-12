import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import boto3
import io


CONFIG_PATH = os.getenv("config_path_analysis")

with open(CONFIG_PATH, "r") as file:
    config = json.load(file)

print(config)
bucket_name = 'renaibucket1.0'

# Create S3 client
s3_client = boto3.client('s3')

# Read master_outcome_df from S3
response = s3_client.get_object(Bucket=bucket_name, Key='master_outcome_df.csv')
df = pd.read_csv(io.BytesIO(response['Body'].read()))

# Filter to choose on series of runs
df = df[(df["training_period_data_size"]==config["training_period_data_size"]) 
        & (df["account_size"]==config["account_size"])
        & (df["risk_per_trade_percent"]==config["risk_per_trade_percent"])
        &  (df["risk_free_annual_return"]==config["risk_free_annual_return"])]


df['Profit'] = df['win'] - df['fee_percent']
df['Time'] = pd.to_datetime(df['Time'])
df['Year'] = df['Time'].dt.year
df['Month'] = df['Time'].dt.month

for r2r in df["risk_to_reward_ratio"].unique():
    # Filter to one r2r for each plot
    r2r_df = df[df['risk_to_reward_ratio']==r2r]
    # Assure no duplicates
    most_rows_per_target = (
        r2r_df.groupby(['target_width', 'run_id'])
        .size()  # Count the number of rows
        .reset_index(name='row_count')
        .sort_values(['target_width', 'row_count'], ascending=[True, False])
        .drop_duplicates('target_width')
    )
    filtered_df = r2r_df.merge(most_rows_per_target[['target_width', 'run_id']], on=['target_width', 'run_id'])
    # Make plot
    grouped = filtered_df.groupby(['Year', 'Month', 'target_width'])['Profit'].sum().reset_index()
    pivot_df = grouped.pivot_table(index=['Year', 'Month'], columns='target_width', values='Profit')
    output_folder = 'analysis/r2r_cw_study/'
    os.makedirs(output_folder, exist_ok=True)
    grouped.to_csv(f'{output_folder}grouped_data.csv', index=False)
    plt.figure(figsize=(8, 5))
    pivot_df.plot(kind='line', marker='o')
    plt.title(f'Profit Over Time by Target Width for r2r: {r2r}')
    plt.xlabel('Time (Year-Month)')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend(title='Target Width')
    # Save the plot
    plt.savefig(f'{output_folder}profit_over_time_r2r_{r2r}.png')
    plt.show()

    print("✅ Analysis saved in 'analysis/r2r_cw_study/'")
