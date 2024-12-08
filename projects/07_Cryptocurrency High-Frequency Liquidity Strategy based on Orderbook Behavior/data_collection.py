import requests
import pandas as pd
import numpy as np
from config import *
import matplotlib.pyplot as plt
import time
from datetime import datetime


class DataCollector:
    def __init__(self):
        self.base_url = BINANCE_BASE_URL
        self.levels = 10
        self.rolling_window_size = 5  # Define the rolling window size
        
    def fetch_orderbook(self, symbol, limit=30):
        params = {
            "symbol": symbol,
            "limit": limit
        }
        response = requests.get(f"{self.base_url}{DEPTH_ENDPOINT}", params=params)
        return response.json()
    
    def calculate_orderbook_features(self, bids, asks):
        """Calculate orderbook features based on the paper's methodology"""
        features = {}

        # Price and volume features
        bid_prices = np.array([float(bid[0]) for bid in bids[:self.levels]])
        bid_volumes = np.array([float(bid[1]) for bid in bids[:self.levels]])
        ask_prices = np.array([float(ask[0]) for ask in asks[:self.levels]])
        ask_volumes = np.array([float(ask[1]) for ask in asks[:self.levels]])
        
        # Mid price
        mid_price = (bid_prices[0] + ask_prices[0]) / 2
        features['mid_price'] = mid_price
        
        # Spread
        features['spread'] = ask_prices[0] - bid_prices[0]
        
        # Order book imbalance
        total_bid_volume = np.sum(bid_volumes)
        total_ask_volume = np.sum(ask_volumes)
        features['volume_imbalance'] = (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
        
        # Price impact features
        features['bid_price_impact'] = np.sum(bid_volumes * (mid_price - bid_prices)) / total_bid_volume
        features['ask_price_impact'] = np.sum(ask_volumes * (ask_prices - mid_price)) / total_ask_volume
        
        # Weighted price features
        features['weighted_bid_price'] = np.sum(bid_prices * bid_volumes) / total_bid_volume
        features['weighted_ask_price'] = np.sum(ask_prices * ask_volumes) / total_ask_volume

        return features
    
    def collect_data(self, symbol=SYMBOL, limit=LIMIT):
        X_data = []
        y_data = []
        rolling_window = []  
        bid_data = []
        ask_data = []

        for _ in range(limit):
            if _ % 100 == 0:
                print(f"Fetching data: {_}/{limit}")

            orderbook = self.fetch_orderbook(symbol)
            bid_data.append(orderbook['bids'])
            ask_data.append(orderbook['asks'])

            features = self.calculate_orderbook_features(orderbook['bids'], orderbook['asks'])

            # Predict the next mid price using the rolling window
            response = features.pop('mid_price')
            predictor = features
            if len(rolling_window) < self.rolling_window_size:
                rolling_window.append(predictor)
            elif len(rolling_window) == self.rolling_window_size:
                X_data.append(rolling_window.copy())
                y_data.append(response)
                rolling_window.pop(0)
                rolling_window.append(predictor)

            df = pd.DataFrame(X_data)
            # Predict future mid price with current rolling features
            df['target'] = pd.Series(y_data).shift(5)
            
            # Save data and plot at checkpoints
            if _ in [2999, 4999, 7999]:
                df.to_csv(f'{DATA_DIR}/{TIMESTAMP}/{_+1}.csv', index=False)
            
            time.sleep(1)  

        pd.DataFrame(bid_data).to_csv(f'{DATA_DIR}/{TIMESTAMP}/raw_bid_data.csv', index=False)
        pd.DataFrame(ask_data).to_csv(f'{DATA_DIR}/{TIMESTAMP}/raw_ask_data.csv', index=False)

        return df
    

    def plot_data(self, parsed_data, data_columns=7):
        # Data columns(7): mid_price, spread, volume_imbalance, bid_price_impact, ask_price_impact, weighted_bid_price, weighted_ask_price
        fig, axes = plt.subplots(data_columns, 1, figsize=(10, 18))
        fig.suptitle("Orderbook Features")

        # Parse each feature dictionary column into separate DataFrames
        
        # Average all entries if requested or plot based on the first row
        average_data = pd.concat(parsed_data).groupby(level=0).mean()
        first_data = pd.concat(parsed_data).iloc[0]  # Just the first row as a DataFrame
        # print(average_data.head())
        # print(first_data.head())

        # Choose which data to plot
        plot_data = average_data  # Set to first_data if only plotting first entry

        for i, feature in enumerate(plot_data.columns):
            axes[i].plot(plot_data.index, plot_data[feature], label=feature)
            axes[i].set_title(feature)
            axes[i].set_xlabel("Index")
            axes[i].set_ylabel(feature)
            axes[i].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f'{PLOTS_DIR}/{TIMESTAMP}/orderbook_features.png')
        plt.close()

        return average_data, first_data
