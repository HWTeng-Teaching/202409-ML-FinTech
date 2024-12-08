import numpy as np
import pandas as pd

class TradingStrategy:
    def __init__(self, model, rolling_window_size=5, threshold=0.001):
        self.model = model
        self.rolling_window_size = rolling_window_size
        self.threshold = threshold
        self.rolling_window = []  # Initialize rolling window

    def update_rolling_window(self, features):
        """Update the rolling window with the latest features."""
        if len(self.rolling_window) < self.rolling_window_size:
            self.rolling_window.append(features)
        else:
            self.rolling_window.pop(0)
            self.rolling_window.append(features)

    def average_rolling_window(self):
        """Calculate the average of features in the rolling window."""
        if len(self.rolling_window) == self.rolling_window_size:
            # Convert list of dicts into a DataFrame and calculate the mean
            rolling_df = pd.DataFrame(self.rolling_window)
            return rolling_df.mean().values
        return None

    def predict_with_rolling_window(self):
        """Use the average of the rolling window for prediction."""
        averaged_features = self.average_rolling_window()
        if averaged_features is not None:
            X_input = averaged_features.reshape(1, -1)  # Reshape for the model
            return self.model.predict(X_input)[0]
        return None

    def generate_signal(self, P_pred, P_current):
        """Generate trading signal based on the prediction and current price."""
        signal = 1 if (P_pred - P_current) / P_current > self.threshold else \
                 -1 if (P_pred - P_current) / P_current < -self.threshold else 0
        return signal

    def backtest(self, data, P_column='price', feature_columns=None):
        """
        Backtest the strategy with rolling window approach.
        
        :param data: DataFrame containing price and features.
        :param P_column: Column name for the current price.
        :param feature_columns: List of feature column names for rolling window.
        """
        if feature_columns is None:
            raise ValueError("feature_columns must be specified for rolling window.")

        # Initialize backtesting results
        signals = []
        profits = []
        rolling_window = []

        for index, row in data.iterrows():
            features = row[feature_columns].to_dict()

            # Update rolling window
            self.update_rolling_window(features)

            # Predict with rolling window
            P_pred = self.predict_with_rolling_window()

            if P_pred is not None:
                # Generate trading signal
                P_current = row[P_column]
                signal = self.generate_signal(P_pred, P_current)

                # Calculate profit
                profit = 0
                if signal == 1:
                    profit = P_pred - P_current  # Long trade
                elif signal == -1:
                    profit = P_current - P_pred  # Short trade

                # Append results
                signals.append(signal)
                profits.append(profit)
            else:
                # Append default values when rolling window is incomplete
                signals.append(0)
                profits.append(0)

        # Add backtesting results to DataFrame
        data['signal'] = signals
        data['profit'] = profits

        # Calculate metrics
        cumulative_returns = (1 + pd.Series(profits)).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        sharpe_ratio = np.mean(profits) / np.std(profits) if np.std(profits) != 0 else 0
        profitability = cumulative_returns.iloc[-1] - 1

        metrics = {
            'Sharpe Ratio': sharpe_ratio,
            'Maximum Drawdown': max_drawdown,
            'Profitability': profitability
        }

        return metrics, data
