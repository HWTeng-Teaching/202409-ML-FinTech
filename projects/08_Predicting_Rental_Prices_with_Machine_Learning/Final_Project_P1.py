# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:17:39 2024

@author: Chang Hung Cheng
"""

import glob
import pandas as pd
import os

# 指定資料夾路徑
folders = ["20240921_opendata", "20240911_opendata", "20240901_opendata", "20240821_opendata", "20240811_opendata"]

# 儲存所有符合條件的檔案
all_files = []
for folder in folders:
    files = glob.glob(f"{folder}/*_lvr_land_c.csv")  # 搜尋結尾為 _lvr_land_c.csv 的檔案
    all_files.extend(files)

# 建立空的 DataFrame 用於合併所有資料
all_data = pd.DataFrame()

for file in all_files:
    # 讀取 CSV 檔案並忽略第一行
    df = pd.read_csv(file).iloc[1:, :]
    df.columns = pd.read_csv(file).iloc[0].tolist()

    # 提取檔名的第一個字母並新增一個欄位
    #file_label = os.path.basename(file)[0]  # 取得檔名第一個字母
    #df['City Code'] = file_label  # 新增欄位並填入檔名標識
    
    # 合併資料
    all_data = pd.concat([all_data, df], ignore_index=True)
    

# 檢查合併後的資料
print(all_data.head())


df_rent = all_data.copy()

rent = df_rent[(df_rent["transaction sign"] == "租賃房屋") & (df_rent["main use"] == "住家用")]
rent.dropna(subset=['construction to complete the years'], inplace = True)

import matplotlib.pyplot as plt

unique = rent.nunique()


plt.figure(figsize=(15, 6))
unique.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Unique Value  in rent DataFrame', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Unique Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




# 計算每個欄位的 NaN 比例
nan_ratio = rent.isna().mean()
threshold = 0.5
rent = rent.loc[:, nan_ratio <= threshold]
rent = rent.drop(rent.columns[[1, 2, 3, 5, 9, 16, 20, 21, 22, 23, 24]], axis=1)


describe = rent.describe()
unique = rent.nunique()



plt.figure(figsize=(15, 6))
unique.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Unique Value  in rent DataFrame', fontsize=16)
plt.xlabel('Columns', fontsize=14)
plt.ylabel('Unique Count', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



rent['transaction year month and day'] = rent['transaction year month and day'].astype(int) + 19110000
rent['transaction year month and day'] = pd.to_datetime(rent['transaction year month and day'], format='%Y%m%d')

rent['construction to complete the years'] = rent['construction to complete the years'].astype(int) + 19110000
rent['construction to complete the years'] = pd.to_datetime(rent['construction to complete the years'], format='%Y%m%d', errors='coerce')

rent = rent.dropna(subset=['construction to complete the years', 'Rental period'])

from datetime import datetime

def roc_to_ad(roc_date_str):
    # 提取年份、月份和日期
    roc_year = int(roc_date_str[:3])  # 民國年份，前三位
    roc_month = int(roc_date_str[3:5])  # 月份，第四和第五位
    roc_day = int(roc_date_str[5:])  # 日期，第六和第七位
    
    # 轉換為西元年份
    ad_year = roc_year + 1911
    
    # 創建西元日期並返回
    return datetime(ad_year, roc_month, roc_day)

# 拆分 'date_range' 列並轉換為西元日期
rent[['start_date', 'end_date']] = rent['Rental period'].str.split('~', expand=True)

# 將民國日期轉換為西元日期
rent['start_date'] = rent['start_date'].apply(roc_to_ad)
rent['end_date'] = rent['end_date'].apply(roc_to_ad)


rent_df = pd.DataFrame()
rent_df["villages/urban_district"] = rent[['The villages and towns urban district']]
rent_df["house_age"] = (rent['transaction year month and day'] - rent['construction to complete the years']).apply(lambda x: x.days / 365)
rent_df["level"] = rent["shifting level"]
rent_df["level_total"] = rent["total floor number"]
rent_df["state"] = rent["building state"]
rent_df["materials"] = rent["main building materials"]
rent_df["area"] = rent["building shifting total area"]
rent_df["bedroom"] = rent['Building present situation pattern - room'].astype(int)
rent_df["livingroom"] = rent['building present situation pattern - hall'].astype(int)
rent_df["bathroom"] = rent['building present situation pattern - health'].astype(int)
rent_df['manage_dummy'] = rent['Whether there is manages the organization'].map({'有': 1, '無': 0})
rent_df['furniture_dummy'] = rent['Whether there is attaches the furniture'].map({'有': 1, '無': 0})
rent_df['elevator_dummy'] = rent['elevator'].map({'有': 1, '無': 0})
rent_df['manager_dummy'] = rent['Residential Manager'].map({'有': 1, '無': 0})
rent_df['type'] = rent['Rental type']
rent_df['equipment_note'] = rent['equipment']
rent_df['services_type'] = rent['Rental residential services']
rent_df['rental_period'] = (rent['end_date']-rent['start_date']).dt.days
rent_df['1F_dummy'] = (rent['shifting level'] == "一層").astype(int)
rent_df["price"] = rent["total price NTD"].astype(int)


rent_df = rent_df[rent_df["rental_period"]<10000]

import seaborn as sns

sns.pairplot(rent_df)
plt.show()



# 選擇數字型資料
numeric_df = rent_df.select_dtypes(include=['number'])

# 繪製相關係數熱圖
plt.figure(figsize=(10, 10))
sns.heatmap(numeric_df.corr(), annot=False, center=0.0, cmap='coolwarm')
plt.show()

rent_df.describe()

rent_df.to_excel("rent_df.xlsx")


