# models.py
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from config import *

class TradingStrategy:
    def __init__(self, model, feature_extractor, threshold=0.001):
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        
    def generate_signal(self, y_pred, y_true, threshold=0.001):
        """Generate trading signal based on the model's prediction"""
        signal = 1 if y_pred > y_true + threshold else -1 if y_pred < y_true - threshold else 0
        return signal
    
class ModelTrainer:
    def __init__(self, X):
        self.models = {
            'Linear Regression': LinearRegression(),
            'XGBoost': self.build_xgboost_model(),
            'SVR': self.build_svr_model(),
            'LSTM': self.build_lstm_model(input_shape=(X.shape[1], X.shape[2])),
        }
        self.metrics = {}
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=25),
            Dense(units=1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def build_xgboost_model(self):
        """Build XGBoost model"""
        return XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    
    def build_svr_model(self):
        """Build SVR model"""
        return SVR(kernel='rbf', C=1.0, epsilon=0.1)
    
    def train_evaluate(self, X_train, X_test, y_train, y_test, X_seq_train, X_seq_test, y_seq_train, y_seq_test):
        results = {}
        
        for name, model in self.models.items():
            # Train model and make predictions
            if name == 'LSTM':
                model.fit(X_seq_train, y_seq_train, epochs=10, batch_size=32)
                y_train_pred = model.predict(X_seq_train)
                y_test_pred = model.predict(X_seq_test)
                results[name] = {
                    'train': self._calculate_metrics(y_seq_train, y_train_pred),
                    'test': self._calculate_metrics(y_seq_test, y_test_pred)
                }
            else:
                model.fit(X_train, y_train)
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                results[name] = {
                    'train': self._calculate_metrics(y_train, y_train_pred),
                    'test': self._calculate_metrics(y_test, y_test_pred)
                }
            
            # Store prediction to csv
            np.savetxt(f'{DATA_DIR}/{name}_train_pred.csv', y_train_pred, delimiter=',')
            np.savetxt(f'{DATA_DIR}/{name}_test_pred.csv', y_test_pred, delimiter=',')
            
        self._plot_results(results)
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        return {
            'Accuracy': self._calculate_signals_accuracy(y_pred, y_true),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }
    
    def _calculate_signals_accuracy(self, y_pred, y_true):
        correct_signals = np.sum((np.sign(y_pred)==np.sign(y_true)) & (np.abs(y_pred - y_true)/np.abs(y_true) < 0.00005))
        print(f'Correct signals: {correct_signals}/{len(y_true)}')
        return correct_signals / len(y_true)
    
    def _plot_results(self, results):
        metrics = ['MSE', 'RMSE', 'MAE', 'R2']
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx//2, idx%2]
            
            data = []
            for model_name in results.keys():
                data.append([
                    results[model_name]['train'][metric],
                    results[model_name]['test'][metric]
                ])
            
            x = np.arange(len(results.keys()))
            width = 0.25
            
            ax.bar(x - width, [d[0] for d in data], width, label='Train')
            ax.bar(x + width, [d[1] for d in data], width, label='Test')
            
            ax.set_title(f'{metric} Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(results.keys())
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{PLOTS_DIR}/model_comparison.png')
        plt.close()
