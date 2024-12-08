# feature_engineering.py
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from config import *
import re
import ast


class FeatureEngineer:
    def __init__(self, sequence_length=60):
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
    
    def parse_feature_column(self, column):
        """
        Parse feature column from string to DataFrame with specific features
        and align it with the target column for prediction.
        """
        parsed_data = []

        for entry in column:
            # Convert numpy float64 to regular floats using regex replacement
            entry = re.sub(r"np\.float64\(([^)]+)\)", r"\1", entry)

            try:
                # Parse the string into a dictionary
                parsed_entry = ast.literal_eval(entry)
                parsed_data.append({
                    'spread': parsed_entry['spread'],
                    'volume_imbalance': parsed_entry['volume_imbalance'],
                    'bid_price_impact': parsed_entry['bid_price_impact'],
                    'ask_price_impact': parsed_entry['ask_price_impact'],
                    'weighted_bid_price': parsed_entry['weighted_bid_price'],
                    'weighted_ask_price': parsed_entry['weighted_ask_price']
                })
            except SyntaxError as e:
                print(f"Syntax error while parsing entry: {entry}\nError: {e}")
            except KeyError as e:
                print(f"Key error while parsing entry: {entry}\nMissing key: {e}")

        return pd.DataFrame(parsed_data)

    def analyze_correlations(self, df, target_column='target'):
        """Analyze feature correlations and importance with a rolling window approach."""
        # Rolling window approach impolented when plotting data
        # Calaulate average correlation matrix and target correlations 
        avg_corr_matrix = df.corr()
        # print(avg_corr_matrix.head())

        # Plot average correlation matrix over all windows
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Feature Correlation Matrix over Rolling Windows')
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/{TIMESTAMP}/average_correlation_matrix.png')
        plt.close()

        # Plot average feature correlations with target over all windows
        avg_target_correlations = avg_corr_matrix[target_column].drop(target_column)
        plt.figure(figsize=(10, 6))
        avg_target_correlations.sort_values().plot(kind='bar')
        plt.title('Feature Correlations with Target over Rolling Windows')
        plt.xticks(rotation=45)
        plt.savefig(f'{PLOTS_DIR}/{TIMESTAMP}/average_target_correlation.png')
        plt.close()

        return avg_corr_matrix, avg_target_correlations
    
    def select_features(self, df, corr_matrix, target_correlations, correlation_threshold=0.1):
        """Select features """
        # Select features with 
    
    def prepare_data(self, data, target_column='target', window_size=5):
        """Prepare data using a rolling window approach."""
        feature_data = pd.concat([self.parse_feature_column(data[str(i)]) for i in range(window_size)], axis=1)
        
        # Rename columns to avoid duplicates
        feature_data.columns = [f"{col}_{i}" for i in range(window_size) 
                                for col in feature_data.columns[:len(feature_data.columns) // window_size]]
        target_data = pd.Series(data[target_column])  

        # Fit scaler and transform features and target
        X_scaled = self.scaler.fit_transform(feature_data) # Shape: (samples, features)
        y_scaled = self.scaler.fit_transform(target_data.values.reshape(-1, 1)).flatten() # Shape: (samples,)

        # Drop rows with NaNs in X_scaled or y_scaled
        non_nan_mask = ~np.isnan(X_scaled).any(axis=1) & ~np.isnan(y_scaled)
        X_scaled = X_scaled[non_nan_mask]
        y_scaled = y_scaled[non_nan_mask]
        
        X_seq = []
        y_seq = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y_scaled[i + self.sequence_length])
        
        X_seq = np.array(X_seq) # Shape: (samples, timesteps, features)
        y_seq = np.array(y_seq) # Shape: (samples,)

        return (X_scaled, y_scaled), (X_seq, y_seq)