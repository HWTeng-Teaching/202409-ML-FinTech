# main.py
from data_collection import DataCollector
from feature_engineering import FeatureEngineer
from models import ModelTrainer
from sklearn.model_selection import train_test_split
import pandas as pd
from config import *
import ast
import re

from strategy import TradingStrategy 
from livetrading import LiveTrading 

def parse_feature_column(column):
    """Parse feature column from string to list of dictionaries"""
    parsed_data = []
    for entry in column:
        # Convert numpy float64 to regular floats using regex replacement
        entry = re.sub(r"np\.float64\(([^)]+)\)", r"\1", entry)
        try:
            parsed_entry = ast.literal_eval(entry)
            parsed_data.append(parsed_entry)
        except SyntaxError as e:
            print(f"Syntax error while parsing entry: {entry}\nError: {e}")
    return pd.DataFrame(parsed_data)

def main():
    # Data collection
    print("Collecting data...")
    collector = DataCollector()
    # data = collector.collect_data()
    # Read data from csv file
    data = pd.read_csv(f'{DATA_DIR}/{TIMESTAMP}/8000.csv')

    print("Plotting data...")
    parsed_data = [parse_feature_column(data[col]) for col in data.columns[:-1]]
    parsed_data.append(data['target'])  
    average_data, first_data = collector.plot_data(parsed_data)

    # Feature engineering
    print("Engineering features...")
    engineer = FeatureEngineer()
    
    # Analyze correlations and select features
    print("Analyzing correlations...")
    correlation_matrix, target_correlations = engineer.analyze_correlations(average_data)
    # TODO: selected_features = engineer.select_features(data, correlation_matrix, target_correlations)

    # Prepare data
    data, lstm_data = engineer.prepare_data(data)
    X_scaled, y = data
    X_seq, y_seq = lstm_data
    # Convert to DataFrame for backtesting
    backtest_df = pd.DataFrame(X_scaled, columns=[f'feature_{i}' for i in range(X_scaled.shape[1])])
    backtest_df['price'] = y

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, shuffle=False
    )
    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, shuffle=False
    )
    
    # Train and evaluate models
    print("Training and evaluating models...")
    # trainer = ModelTrainer(X_seq)
    trainer = ModelTrainer(X_seq)
    results = trainer.train_evaluate(X_train, X_test, y_train, y_test, X_seq_train, X_seq_test, y_seq_train, y_seq_test)
    best_model = trainer.models['Linear Regression']  # Replace with logic to choose the best model
    # Save results
    pd.DataFrame(results).to_csv(f'{DATA_DIR}/model_results.csv')

    print("Backtesting strategy...")
    feature_columns = [f'feature_{i}' for i in range(X_scaled.shape[1])]
    strategy = TradingStrategy(best_model, rolling_window_size=5, threshold=0.001)
    metrics, backtest_data = strategy.backtest(backtest_df, P_column='price', feature_columns=feature_columns)
    print("Backtest Metrics:", metrics)

    print("Initializing live trading...")
    live_trading = LiveTrading(strategy, data_logger_path="mock_trading_log.csv")
    live_trading.start_mock_trading(duration_minutes=5)

    print("Process completed successfully!")

if __name__ == "__main__":
    main()
