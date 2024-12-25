# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 03:33:49 2024

@author: Chang Hung Cheng
"""

import pandas as pd
import numpy as np
#from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 讀取數據
train_data = pd.read_excel("rent_train_df.xlsx")
#validation_data = pd.read_excel("rent_validation_df.xlsx")
test_data = pd.read_excel("rent_test_df.xlsx")

# 查看數據基本信息
print("訓練數據預覽：\n", train_data.head())
#print("驗證數據預覽：\n", validation_data.head())
print("測試數據預覽：\n", test_data.head())

# 數據預處理（假設數據已清理）
# 確定特徵與目標變數
target = 'price'  # 假設 price 為目標變數
features = [col for col in train_data.columns if col != target]

# 分離特徵與目標
X_train, y_train = train_data[features], train_data[target]
#X_val, y_val = validation_data[features], validation_data[target]
X_test, y_test = test_data[features], test_data[target]

# 儲存模型結果
results = {}

# 定義模型清單
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=100)
}


# 確認特徵類型
print(train_data.dtypes)

# 找出文字型（類別型）特徵
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()
print("類別型特徵:", categorical_features)

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# 確定數值型和類別型特徵
target = 'price'  # 假設目標變數是 'price'
numerical_features = train_data.select_dtypes(exclude=['object']).columns.tolist()
numerical_features.remove(target)  # 移除目標變數
categorical_features = train_data.select_dtypes(include=['object']).columns.tolist()

# 建立編碼處理器
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # 保留數值特徵
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # 編碼類別特徵
    ]
)

# 處理數據
X_train = preprocessor.fit_transform(train_data.drop(columns=[target]))
#X_val = preprocessor.transform(validation_data.drop(columns=[target]))
X_test = preprocessor.transform(test_data.drop(columns=[target]))

# 提取目標變數
y_train = train_data[target].values
#y_val = validation_data[target].values
y_test = test_data[target].values



from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 設定超參數搜尋空間
param_grids = {
    'Linear Regression': {},
    'Decision Tree': {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 10, 20]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 10]
    }
}

# 儲存最佳模型和結果
best_models = {}
results = {}

# 針對每個模型進行超參數調整與評估
for name, model in models.items():
    print(f"正在調整模型: {name}")
    if name in param_grids and param_grids[name]:  # 若模型有超參數
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            scoring='neg_mean_squared_error', 
            cv=5, 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"{name} 最佳超參數: {best_params}")
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = "無"
    
    # 測試集評估
    y_test_pred = best_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 儲存結果
    best_models[name] = best_model
    results[name] = {
        'Best Params': best_params,
        'Test MSE': test_mse,
        'Test MAE': test_mae,
        'Test R²': test_r2
    }

    print(f"{name} 測試結果: MSE={test_mse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.2f}\n")

# 輸出完整結果
print("所有模型結果：")
for model_name, metrics in results.items():
    print(f"\n模型: {model_name}")
    for metric_name, value in metrics.items():
        if metric_name == 'Best Params':
            print(f"{metric_name}: {value}")
        else:
            print(f"{metric_name}: {value:.2f}")

# 保存所有模型的預測結果
predicted_test_data = test_data.copy()
for name, model in best_models.items():
    predicted_test_data[f'{name}_Predicted_Price'] = model.predict(X_test)

predicted_test_data.to_excel("rent_test_predictions_optimized.xlsx", index=False)
print("所有最佳模型測試集預測結果已保存至 `rent_test_predictions_optimized.xlsx`")


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


models = ['Linear Regression_Predicted_Price', 'Decision Tree_Predicted_Price', 'Random Forest_Predicted_Price', 'XGBoost_Predicted_Price']
errors = {model: (predicted_test_data[model] - predicted_test_data['price'])**2 for model in models}

# 使用logspace設置對數刻度的bins範圍
bin_range = np.logspace(1, 10, num=20)

# 創建一個圖形，並且讓所有子圖共享Y軸
fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharey=True)


for i, model in enumerate(models):
    ax = axes[i // 2, i % 2]  # 選擇子圖位置
    data = errors[model]

    # 計算直方圖
    freq, bin_edges = np.histogram(data, bins=bin_range)
    cumulative_freq = np.cumsum(freq)  # 累積次數
    total_count = cumulative_freq[-1]  # 總次數

    # 根據累積次數找到分位點
    p10_idx = np.searchsorted(cumulative_freq, total_count * 0.10)
    q1_idx = np.searchsorted(cumulative_freq, total_count * 0.25)
    q2_idx = np.searchsorted(cumulative_freq, total_count * 0.50)  # 中位數
    q3_idx = np.searchsorted(cumulative_freq, total_count * 0.75)
    p90_idx = np.searchsorted(cumulative_freq, total_count * 0.90)

    # 將分位點索引轉換為邊界值
    p10 = bin_edges[p10_idx]
    q1 = bin_edges[q1_idx]
    q2 = bin_edges[q2_idx]
    q3 = bin_edges[q3_idx]
    p90 = bin_edges[p90_idx]

    # 繪製直方圖
    ax.hist(data, bins=bin_range, color='skyblue', edgecolor='black', alpha=0.7)
    
    # 繪製分位數
    ax.axvline(p10, linestyle='--', color='red', label=f'P10: {p10:.2f}')
    ax.axvline(q1, linestyle='--', color='green', label=f'Q1: {q1:.2f}')
    ax.axvline(q2, linestyle='--', color='blue', label=f'Q2 : {q2:.2f}')
    ax.axvline(q3, linestyle='--', color='purple', label=f'Q3: {q3:.2f}')
    ax.axvline(p90, linestyle='--', color='orange', label=f'P90: {p90:.2f}')
    
    # 設置X軸為對數刻度
    ax.set_xscale('log')
    ax.set_title(f'{model} Error Distribution (Log scale)')
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 顯示圖例
    ax.legend()

# 調整子圖間距並顯示圖表
plt.tight_layout()
plt.show()