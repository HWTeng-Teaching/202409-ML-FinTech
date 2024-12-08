import websocket
import json
import pandas as pd
from strategy import TradingStrategy
from datetime import datetime, timedelta
import numpy as np
import time
from config import SYMBOL

class LiveTrading:
    def __init__(self, strategy, data_logger_path="mock_trading_log.csv"):
        self.strategy = strategy
        self.ws_url = f"wss://stream.binance.com:9443/ws/{SYMBOL.lower()}@depth10"
        self.data_logger_path = data_logger_path
        self.trading_data = []
        self.stop_time = None

    def on_message(self, ws, message):
        data = json.loads(message)
        bids = np.array([float(bid[0]) for bid in data['bids'][:self.strategy.rolling_window_size]])
        asks = np.array([float(ask[0]) for ask in data['asks'][:self.strategy.rolling_window_size]])
        bid_volumes = np.array([float(bid[1]) for bid in data['bids'][:self.strategy.rolling_window_size]])
        ask_volumes = np.array([float(ask[1]) for ask in data['asks'][:self.strategy.rolling_window_size]])
        
        # Extract features
        mid_price = (bids[0] + asks[0]) / 2
        features = {
            'spread': asks[0] - bids[0],
            'volume_imbalance': (bid_volumes.sum() - ask_volumes.sum()) / (bid_volumes.sum() + ask_volumes.sum()),
            'bid_price_impact': np.sum(bid_volumes * (mid_price - bids)) / bid_volumes.sum(),
            'ask_price_impact': np.sum(ask_volumes * (asks - mid_price)) / ask_volumes.sum(),
            'weighted_bid_price': np.sum(bids * bid_volumes) / bid_volumes.sum(),
            'weighted_ask_price': np.sum(asks * ask_volumes) / bid_volumes.sum()
        }
        
        # Update rolling window
        self.strategy.update_rolling_window(features)

        # Make prediction
        P_pred = self.strategy.predict_with_rolling_window()
        if P_pred is not None:
            signal = self.strategy.generate_signal(P_pred, mid_price)
            profit = self.mock_trade(signal, mid_price, P_pred)
            print(f"Signal: {signal}, Current Price: {mid_price}, Predicted Price: {P_pred}, Profit: {profit}")

        # Stop trading after the specified time
        if datetime.now() >= self.stop_time:
            ws.close()

    def mock_trade(self, signal, price, predicted_price):
        """Simulate a trade and calculate mock profit."""
        trade_data = {
            "timestamp": datetime.now().isoformat(),
            "signal": signal,
            "current_price": price,
            "predicted_price": predicted_price,
            "profit": 0  # Default profit
        }

        if signal == 1:
            trade_data["profit"] = predicted_price - price
        elif signal == -1:
            trade_data["profit"] = price - predicted_price

        self.trading_data.append(trade_data)
        self.save_mock_trades()
        return trade_data["profit"]

    def save_mock_trades(self):
        """Save mock trading data to a CSV file."""
        df = pd.DataFrame(self.trading_data)
        df.to_csv(self.data_logger_path, index=False)

    def start_mock_trading(self, duration_minutes=5):
        """
        Start the WebSocket connection for mock trading.
        
        :param duration_minutes: Total time to run the trading in minutes.
        """
        self.stop_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        # Start WebSocket connection
        ws = websocket.WebSocketApp(self.ws_url, on_message=self.on_message)

        # Run the WebSocket with a custom loop for periodic trade execution
        while datetime.now() < self.stop_time:
            ws.run_forever()
            time.sleep(1)  # Delay for 1 second between trades
