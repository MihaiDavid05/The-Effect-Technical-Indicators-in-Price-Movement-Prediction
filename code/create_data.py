import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import os
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt
from create_features import add_MA_indicators, add_OBV_indicator, add_previous_day_volume_indicator, \
                            add_ATR_indicator, add_ATR_indicator, add_AD_indicator, add_MACD_indicators, \
                            add_stochastic_oscillator_indicator, add_MFI_indicator, add_RSI_indicator, \
                            add_target_column, add_log_returns


def get_best_stock_names(url):
    """
    Get top 100 most active stocks from specific url
    """
    req = requests.get(url)
    soup = BeautifulSoup(req.content, 'html.parser')
    table = soup.find("table",{"class":"table-DR3mi0GH"})
    rows = table.findChildren(['tr'])
    stocks = [row.get('data-rowkey') for row in rows if row.get('data-rowkey') is not None]
    stocks = [tuple(s.split(':')) for s in stocks]
    return pd.DataFrame(stocks, columns=['Exchange', 'Name'])


def get_stock_data(raw_data_path, website_url):
    """
    Create dataframe with stock information and save it.
    """
    if not os.path.exists(raw_data_path):
        stock_names_df = get_best_stock_names(website_url)
        raw_df = pd.DataFrame()
        for i, stock in tqdm(enumerate(stock_names_df['Name'].values)):
            msft = yf.Ticker(stock)
            hist = msft.history(period="max").reset_index()
            hist['Name'] = stock
            raw_df = pd.concat([raw_df, hist])
        raw_df['Date'] = raw_df['Date'].dt.strftime('%Y/%m/%d')
        raw_df['Date'] = pd.to_datetime(raw_df['Date'], errors='coerce')
        if 'Adj Close' in raw_df.columns:
            raw_df.drop(['Adj Close'], axis=1, inplace=True)
        raw_df['year'] = raw_df['Date'].dt.year
        raw_df['month'] = raw_df['Date'].dt.month
        raw_df['day'] = raw_df['Date'].dt.day
        
        # Write data to csv
        raw_df.to_csv(raw_data_path, index=False)
    else:
        raw_df = pd.read_csv(raw_data_path)
    
    return raw_df


def convert_size(size_bytes):
    """
    Compute the size of the dataframe used.
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)

    return "%s %s" % (s, size_name[i])


def get_processed_data(processed_data_path, raw_df):
    """
    Add features and target columns to raw data and save it.
    """
    if not os.path.exists(processed_data_path):
        grouped_df = raw_df.groupby('Name')
        new_groups = []
        for name, group in tqdm(grouped_df):
            # Trend indicators
            new_group = add_MA_indicators(group)

            # Volume indicators
            new_group = add_OBV_indicator(new_group)
            new_group = add_AD_indicator(new_group)
            new_group = add_previous_day_volume_indicator(new_group)

            # Volatility indicators
            new_group = add_ATR_indicator(new_group)

            # Momentum indicators
            new_group = add_MACD_indicators(new_group)
            new_group = add_stochastic_oscillator_indicator(new_group)
            new_group = add_MFI_indicator(new_group)
            new_group = add_RSI_indicator(new_group)

            # Target column
            new_group = add_target_column(new_group)

            # Log-returns
            new_group = add_log_returns(new_group)

            new_groups.append(new_group)

        df = pd.concat(new_groups)
        df = df.dropna()
        print(f'DataFrame shape before adding features: {raw_df.shape}')
        print(f'DataFrame shape after adding features and removing NaNs: {df.shape}')
        df.to_csv(processed_data_path)
    else:
        df = pd.read_csv(processed_data_path)

    return df


def prepare_for_clustering(df, show=False):
    """
    Prepare data for clustering.
    """
    df_cluster = df.sort_values('Date')[['Date', 'Close', 'Name']]
    df_cluster = df_cluster.set_index(['Date','Name'])
    df_cluster = df_cluster.unstack()['Close']
    df_cluster = df_cluster.reset_index()

    # Compute log returns
    df_cluster = np.log(df_cluster.iloc[:, 1:]).diff()

    if show:
        # Plot data
        fig, ax1 = plt.subplots(figsize=(20, 15))
        df_cluster.plot(ax=ax1, legend=False)
        plt.tight_layout()
        plt.show()
    
    return df_cluster


