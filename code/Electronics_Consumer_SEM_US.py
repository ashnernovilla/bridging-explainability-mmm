# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import Library

# %%
# ---------------------------------------------------------
# Standard Library Imports
# ---------------------------------------------------------
import os
import time
import random
import warnings
import traceback
from datetime import datetime

# ---------------------------------------------------------
# Core Data & Computation
# ---------------------------------------------------------
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Machine Learning & Statistics
# ---------------------------------------------------------
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import (
    r2_score, 
    mean_absolute_error, 
    root_mean_squared_error, 
    mean_absolute_percentage_error
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.seasonal import seasonal_decompose
import semopy
from semopy import Model
import arviz as az

# ---------------------------------------------------------
# Data Visualization
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# External Data & APIs
# ---------------------------------------------------------
import yfinance as yf
import holidays
from fredapi import Fred
from pytrends.request import TrendReq
from trendspy import Trends

# ---------------------------------------------------------
# Web & System Utilities
# ---------------------------------------------------------
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import psutil
import IPython
from IPython.display import Image, display

# ---------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------
# Pandas Display Settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

# Default Visualization Theme
sns.set_theme()

# Hardware Check
ram_gb = psutil.virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

# %% [markdown]
# # Import File

# %%
# ------------------------------------------------------------------------------
# 1. LOAD & CLEAN
# ------------------------------------------------------------------------------
df_raw = pd.read_excel(r'../data/consumer_electronics_data.xlsx', sheet_name='Raw Data')

# Clean Column Names & Currency
df_raw.columns = df_raw.columns.str.strip()

def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('$', '').replace(',', '')
        if '(' in x and ')' in x: x = x.replace('(', '-').replace(')', '')
        try: return float(x)
        except: return 0.0
    return x

df_raw['Spend'] = df_raw['Spend'].apply(clean_currency)

# Clean Date & Normalize to Monday
df_raw['Week'] = pd.to_datetime(df_raw['Week'])
df_raw['Week'] = df_raw['Week'] - pd.to_timedelta(df_raw['Week'].dt.dayofweek, unit='D')

df_raw.rename(columns={'ï»¿Country':'Country'}, inplace=True)

df_raw.head()

# %% [markdown]
# # Exploratory Data Analysis

# %%
display(df_raw.head())
display(df_raw.tail())

# %% [markdown]
# ### Data Transformation

# %%
df = df_raw.copy()

df['Country'] = np.where(df['Country']=='US/CA', 'US', df['Country'])

# Filter Country (Keep US/CA)
if 'Country' in df.columns:
    df = df[df['Country'] == 'US']

# ------------------------------------------------------------------------------
# 2. HYBRID GROUPING LOGIC (Base Channel + Refinements)
# ------------------------------------------------------------------------------
def get_hybrid_group(row):
    # Start with the existing Base Channel as the default
    base = str(row['Base Channel']).strip()
    chan = str(row['Channel']).strip().upper()
    
    # --- EXCEPTION 1: TARGET VARIABLE ---
    if 'SALES' in base.upper() and 'DATA' in base.upper():
        return "SalesData"

    # --- EXCEPTION 2: VIDEO SPLITS ---
    # We check if the Base Channel indicates Programmatic/Video/Display, then refine.
    if any(x in base.upper() for x in ['PROGRAMMATIC', 'VIDEO', 'CTV', 'DISPLAY']):
        # Determine Business Unit (ACP vs ECOMM) from the Base Channel string
        
        # --- EXCEPTION 2: VIDEO / DISPLAY / AUDIO (Unify ACP + ECOMM) ---
                
        # 1. YouTube (Explicit)
        if 'YOUTUBE' in chan:
            return "Video_YouTube"
        
        # 2. CTV (Explicit in Channel OR Base)
        if 'CTV' in chan or 'CTV' in base.upper():
            return "Video_CTV"
            
        # 3. OLV (Online Video - if not YouTube/CTV but contains 'Video')
        if 'VIDEO' in chan or 'VIDEO' in base.upper():
            return "Video_OLV"
        
        # 4. Audio (Audio - if not YouTube/CTV/OLV but contains 'Audio')
        if 'AUDIO' in chan or 'AUDIO' in base.upper():
            return "Audio"
            

    # --- EXCEPTION 3: PAID SOCIAL SPLITS ---
    # Base Channel usually just says "ACP_Paid Social". We want to know if it's Meta or LinkedIn.
    if 'SOCIAL' in base.upper() and 'PAID' in base.upper():
        
        # --- EXCEPTION 3: PAID SOCIAL (Unify ACP + ECOMM) ---
        
        if any(x in chan for x in ['FACEBOOK', 'INSTAGRAM', 'META']): 
            return "PaidSocial_Meta"
        
        if 'LINKEDIN' in chan: 
            return "PaidSocial_LinkedIn"
        
        if 'TIKTOK' in chan: 
            return 'PaidSocial_TikTok'
        
        if 'PINTEREST' in chan:
            return 'PaidSocial_Pinterest'
        
        if 'SNAPCHAT' in chan: 
            return 'PaidSocial_Snapchat'
        
        if 'REDDIT' in chan: 
            return 'PaidSocial_Reddit'
        
        return f"PaidSocial_Other" 

    # --- EXCEPTION 4: EMAIL / WUNDERKIND ---
    # Base Channel "WunderkindEmail" -> Map to just "Email" for simplicity
    if 'WUNDERKIND' in base.upper() or 'EMAIL' in base.upper():
        return "Email"
        
    # --- DEFAULT: USE BASE CHANNEL ---
    # For Paid Search, Direct Mail, Affiliate, Organic -> The Base Channel is already perfect.
    # We just clean it up (replace spaces with underscores) to be safe.
    clean_base = base.replace('ACP', '').replace('ECOMM', '').strip()
    # return clean_base.replace(' ', '_').replace('/', '_').replace('__', '_')
    return clean_base.replace(' ', '_').replace('/', '_')

    # return base.replace(' ', '_').replace('/', '_')

df['Final_Group'] = df.apply(get_hybrid_group, axis=1)

# ------------------------------------------------------------------------------
# 3. AGGREGATE & PIVOT
# ------------------------------------------------------------------------------
metrics = ['Spend', 'Clicks', 'Impressions', 'Opens', 'Circulations', 'Sales Revenue']
existing_metrics = [c for c in metrics if c in df.columns]

# ADD 'Country' to the grouping keys
grouped = df.groupby(['Country', 'Week', 'Final_Group'])[existing_metrics].sum().reset_index()

# Pivot (Index now includes Country)
pivoted = grouped.pivot(index=['Country', 'Week'], columns='Final_Group', values=existing_metrics)

# Flatten Columns
pivoted.columns = [f"{col[0]}_{col[1]}" for col in pivoted.columns]
pivoted = pivoted.reset_index().fillna(0)

# ------------------------------------------------------------------------------
# 4. SELECT COLUMNS FOR MODELING
# ------------------------------------------------------------------------------
# Define the Target
target_col = 'Sales Revenue_SalesData'

cols_to_keep = ['Country', 'Week']
if target_col in pivoted.columns:
    cols_to_keep.append(target_col)

for col in pivoted.columns:
    if col in cols_to_keep: continue
    
    metric, group = col.split('_', 1)
    
    # Search / Affiliate / Email -> Clicks are best volume
    if any(x in group.upper() for x in ['SEARCH', 'AFFILIATE', 'EMAIL']):
        if metric in ['Spend', 'Clicks']: cols_to_keep.append(col)
        
    # Direct Mail -> Circulations
    elif 'DIRECTMAIL' in group.upper():
        if metric in ['Spend', 'Circulations']: cols_to_keep.append(col)
        
    # Display / Video / Social -> Impressions are best volume
    else: 
        if metric in ['Spend', 'Impressions']: cols_to_keep.append(col)

final_df = pivoted[cols_to_keep].copy()

# Print output check
print("Final Shape:", final_df.shape)
# print("Columns Sample:", final_df.columns.tolist()[:10])

final_df.columns = final_df.columns.str.replace('__', '_', regex=False)

display(final_df.head())


# %% [markdown]
# ### SEM Application

# %%
def consolidate_final_spend_columns(df):
    df = df.copy()
    
    # 1. SUM: BRAND SEARCH
    # Navigational/Transactional: The user already knows you. They are just trying to find your specific product to buy it. Example: "Lenovo Laptop," "Lenovo ThinkPad Price"
    # High Reliability: This almost always has high ROAS because the user already wants you. Managers rarely panic here.
    brand_cols = [
        'Spend_Paid_Search_Brand', 'Spend_Paid_Search_Search_-_Brand', 
        'Spend_Paid_Search_Search_-_Core_Brand', 'Spend_Paid_Search_PMAX-Brand', 
        'Spend_Paid_Search_Shopping_-_Brand'
    ]
    # Sum only existing columns
    df['Total_Spend_Search_Brand'] = df[[c for c in brand_cols if c in df.columns]].sum(axis=1)

    # 2. SUM: NON-BRAND SEARCH
    # Informational/Discovery: The user has a problem but doesn't know which brand to choose yet. Example: "Best Business Laptop," "Laptop for Coding," "Windows Laptop"
    nb_cols = [
        'Spend_Paid_Search_Non-Brand', 'Spend_Paid_Search_Search_-_Nonbrand', 
        'Spend_Paid_Search_PMAX-NonBrand', 'Spend_Paid_Search_DSA', 
        'Spend_Paid_Search_Demand_Gen', 'Spend_Paid_Search_Audience', 
        'Spend_Paid_Search_Shopping_-_Nonbrand', 'Spend_Paid_Search_', 'Spend_Paid_Search_PMAX'
    ]
    df['Total_Spend_Search_NonBrand'] = df[[c for c in nb_cols if c in df.columns]].sum(axis=1)

    # 3. SUM: SHOPPING (Generic)
    shop_cols = [
        'Spend_Paid_Search_Shopping', 'Spend_Paid_Search_Shopping_-_Combined'
    ]
    df['Total_Spend_Search_Shopping'] = df[[c for c in shop_cols if c in df.columns]].sum(axis=1)

    # 4. SUM: PROGRAMMATIC DISPLAY
    # Marketing Managers buy software, and the software buys the ads.
    """
    This is the process known as Real-Time Bidding (RTB). It happens in the 200 milliseconds it takes for a webpage to load.
    The User Visits: You click on a website (e.g., CNN.com).
    The Signal: While the page loads, CNN sends a signal to an Ad Exchange: "I have a user here. Who wants to show them an ad?"
    The AI Analysis: Lenovo's DSP (the AI) analyzes your cookie data instantly: "This user looked at laptops yesterday."
    The Bid: The AI calculates: "This user is valuable. I bid $0.05 for this slot."The Win: If Lenovo's AI bids higher than HP's AI, Lenovo wins.
    The Format: The AI checks what kind of slot it won.Is it a square box? -> It inserts a Banner.Is it a headline in the news feed? ->It inserts a Native Ad.Is it a sidebar image? -> It inserts a Static Ad.
    """
    
    prog_cols = [c for c in df.columns if 'Spend_Programmatic_' in c]
    df['Total_Spend_Display_Programmatic'] = df[prog_cols].sum(axis=1)

    # 5. RENAME: SOCIAL & VIDEO (These are already single columns, just renaming for consistency)
    rename_map = {
        'Spend_PaidSocial_Meta': 'Total_Spend_Social_Meta',
        'Spend_PaidSocial_LinkedIn': 'Total_Spend_Social_LinkedIn',
        'Spend_PaidSocial_TikTok': 'Total_Spend_Social_TikTok',
        'Spend_PaidSocial_Pinterest': 'Total_Spend_Social_Pinterest',
        'Spend_PaidSocial_Snapchat': 'Total_Spend_Social_Snapchat',
        'Spend_PaidSocial_Reddit': 'Total_Spend_Social_Reddit',
        'Spend_Video_YouTube': 'Total_Spend_Video_YouTube',
        'Spend_Video_CTV': 'Total_Spend_Video_CTV',
        'Spend_Video_OLV': 'Total_Spend_Video_OLV',
        'Spend_Audio': 'Total_Spend_Audio',
        'Spend_Affiliate': 'Total_Spend_Affiliate',
        'Spend_DirectMail': 'Total_Spend_DirectMail',
        'Spend_Email': 'Total_Spend_Email',
        'Sales Revenue_SalesData': 'Total_Revenue'
    }
    df.rename(columns=rename_map, inplace=True)

    
    # --- 2. ORGANIC MEDIA (USE ACTIVITY) ---
    # The CFO needs to know this value exists, even if it costs $0.
    
    # EMAIL (Use Clicks)
    # Look for 'Clicks_Email' or 'Clicks__Email'
    email_col = [c for c in df.columns if 'Clicks' in c and 'Email' in c]
    print(email_col)
    if email_col:
        df['Organic_Email'] = df[email_col].sum(axis=1)
    else:
        df['Organic_Email'] = 0

    # ORGANIC SOCIAL (Use Impressions)
    # Look for 'Impressions_Organic'
    org_soc_col = [c for c in df.columns if 'Impressions' in c and 'Organic' in c]
    print(org_soc_col)
    if org_soc_col:
        df['Organic_Social'] = df[org_soc_col].sum(axis=1)
    else:
        df['Organic_Social'] = 0
    
    
    # 6. SELECT FINAL LIST
    final_cols = ['Country', 'Week', 'Total_Revenue'] + \
                 ['Total_Spend_Search_Brand', 'Total_Spend_Search_NonBrand', 'Total_Spend_Search_Shopping', 'Total_Spend_Display_Programmatic'] + \
                 list(rename_map.values()) + ['Organic_Email', 'Organic_Social']
    
    # Remove duplicates (in case rename map overlaps with list) and ensure existence
    final_cols = list(set([c for c in final_cols if c in df.columns]))
    
    return df[final_cols]

# Run it
sem_df = consolidate_final_spend_columns(final_df)

sem_df = sem_df[['Week', 'Country', 'Total_Revenue', 'Total_Spend_DirectMail', 'Total_Spend_Social_Snapchat', 'Total_Spend_Social_Reddit', 'Total_Spend_Search_Shopping', 'Total_Spend_Affiliate', 'Total_Spend_Search_Brand', 'Total_Spend_Social_TikTok', 'Total_Spend_Social_Pinterest', 'Total_Spend_Video_CTV', 'Total_Spend_Social_Meta', 'Total_Spend_Social_LinkedIn', 'Total_Spend_Search_NonBrand', 'Total_Spend_Video_OLV', 'Total_Spend_Display_Programmatic', 'Total_Spend_Email', 'Total_Spend_Video_YouTube', 'Total_Spend_Audio', 'Organic_Email', 'Organic_Social']]

print(sem_df.columns.tolist())

display(sem_df.head())

final_df = sem_df.copy()

# The ACM-Ready Variable Mapping
rename_dict = {
    # 1. Identifiers & Target
    'Week': 'Week',
    'Country': 'Country',
    'Total_Revenue': 'Revenue',
    
    # 2. Search Marketing (Keep 'Search' to distinguish from Display/Social)
    'Total_Spend_Search_Brand': 'Spend_Search_Brand',
    'Total_Spend_Search_NonBrand': 'Spend_Search_Generic', # 'Generic' is the academic standard term
    'Total_Spend_Search_Shopping': 'Spend_Shopping',
    
    # 3. Social Media (Drop 'Social_' as it is universally understood)
    'Total_Spend_Social_Meta': 'Spend_Meta',
    'Total_Spend_Social_LinkedIn': 'Spend_LinkedIn',
    'Total_Spend_Social_Reddit': 'Spend_Reddit',
    'Total_Spend_Social_TikTok': 'Spend_TikTok',
    'Total_Spend_Social_Pinterest': 'Spend_Pinterest',
    'Total_Spend_Social_Snapchat': 'Spend_Snapchat',
    
    # 4. Video & Audio (Drop 'Video_' where obvious)
    'Total_Spend_Video_YouTube': 'Spend_YouTube',
    'Total_Spend_Video_CTV': 'Spend_CTV',
    'Total_Spend_Video_OLV': 'Spend_OLV',
    'Total_Spend_Audio': 'Spend_Audio',
    
    # 5. Other Paid Channels
    'Total_Spend_Display_Programmatic': 'Spend_Display',
    'Total_Spend_Affiliate': 'Spend_Affiliate',
    'Total_Spend_DirectMail': 'Spend_Direct_Mail',
    'Total_Spend_Email': 'Spend_Paid_Email', # To distinguish from Organic
    
    # 6. Organic Channels
    'Organic_Email': 'Organic_Email',
    'Organic_Social': 'Organic_Social'
}

# Apply to your dataframe
final_df = final_df.rename(columns=rename_dict)

print("Dataframe successfully renamed for academic presentation.")

del(sem_df)


# %% [markdown]
# ### Spand and Revenue Analysis

# %%
for i in final_df['Country'].unique():
    
    contribution_df = final_df.loc[final_df['Country'] == i]
    
    final_mmm_dataset_columns = pd.DataFrame(contribution_df.columns, columns=['Columns'])
    final_mmm_dataset_columns_sales = final_mmm_dataset_columns.loc[final_mmm_dataset_columns['Columns'].str.contains('Spend_')]
    final_mmm_dataset_columns_sales = final_mmm_dataset_columns_sales['Columns'].tolist()

    final_mmm_dataset_columns_impressions = final_mmm_dataset_columns.loc[final_mmm_dataset_columns['Columns'].str.contains('Impressions_')]
    final_mmm_dataset_columns_impressions = final_mmm_dataset_columns_impressions['Columns'].tolist()

    final_mmm_dataset_columns_clicks = final_mmm_dataset_columns.loc[final_mmm_dataset_columns['Columns'].str.contains('Clicks_')]
    final_mmm_dataset_columns_clicks = final_mmm_dataset_columns_clicks['Columns'].tolist()

    # Define Column Groups
    # ----------------------------
    # 1. Spend Share Analysis
    # ----------------------------
    print(f"\n******** Spend Share Analysis {i} ********")

    if final_mmm_dataset_columns_sales:
        # Calculate total spend across all identified media channels.
        total_media_spend = contribution_df[final_mmm_dataset_columns_sales].sum().sum()
        
        # Compute spend share for each media channel.
        spend_share = contribution_df[final_mmm_dataset_columns_sales].sum() / total_media_spend
        spend_share_df = spend_share.reset_index()
        spend_share_df.columns = ['Media_Channel', 'Spend_Share']
        
        # Sort for plotting
        spend_share_df = spend_share_df.sort_values(by='Spend_Share', ascending=False)
        
        # Display top 10 in text
        print(spend_share_df.head(30))

        # Plot a bar chart of spend share (Top 40 to keep it readable).
        plt.figure(figsize=(10, 16)) # Taller plot for many variables
        top_n = 500
        sns.barplot(x='Spend_Share', y='Media_Channel', 
                    data=spend_share_df.head(top_n))
        plt.title(f"Spend Share by Media Channel (Top {top_n})")
        plt.xlabel("Spend Share")
        plt.ylabel("Media Channel")
        plt.tight_layout()
        plt.show()
    else:
        print("No media channels available for spend share analysis.")

# %%
for i in final_df['Country'].unique():
    
    print(f'====================== {i} =========================')
    
    contribution_df = final_df.loc[final_df['Country'] == i]
    
    final_mmm_dataset_columns = pd.DataFrame(contribution_df.columns, columns=['Columns'])
    final_mmm_dataset_columns_sales = final_mmm_dataset_columns.loc[final_mmm_dataset_columns['Columns'].str.contains('Spend_')]
    final_mmm_dataset_columns_sales = final_mmm_dataset_columns_sales['Columns'].tolist()

    spend_cols = final_mmm_dataset_columns_sales

    # 3. Calculate Global Metrics
    total_media_spend = contribution_df[spend_cols].sum()
    
    print(total_media_spend)


# %% [markdown]
# ## Perform Data Mapping

# %%
spends_mapping = {
    # --- PAID MEDIA (SPEND) ---
    'Spend_Search_Brand':   'Search Brand',
    'Spend_Search_Generic': 'Search Generic', # Updated to match ACM terminology
    'Spend_Shopping':       'Search Shopping',
    'Spend_Meta':           'Social Meta',
    'Spend_LinkedIn':       'Social LinkedIn',
    'Spend_TikTok':         'Social TikTok',
    'Spend_Snapchat':       'Social Snapchat',
    'Spend_Reddit':         'Social Reddit',
    'Spend_Pinterest':      'Social Pinterest',
    'Spend_YouTube':        'Video YouTube',
    'Spend_CTV':            'Video CTV',
    'Spend_OLV':            'Video OLV',
    'Spend_Display':        'Display Programmatic',
    'Spend_Audio':          'Audio',
    'Spend_Affiliate':      'Affiliate',
    'Spend_Direct_Mail':    'Direct Mail', 
}

MEDIA_COLS = list(spends_mapping.keys())

ORGANIC_COLS = [c for c in final_df if 'Organic' in c]

CONTROL_COLS = []
 
data_df = final_df.copy()

SALES_COL = "Revenue"
 
DATE_COL = "Week"

GEO='Country'
  
data_df = data_df[[DATE_COL, GEO, SALES_COL, *MEDIA_COLS ,*CONTROL_COLS, *ORGANIC_COLS]].reset_index(drop=True)


data_df[DATE_COL] = pd.to_datetime(data_df[DATE_COL])

data_df.sort_values(by=[DATE_COL], inplace=True)
   
 
data_df[DATE_COL] = pd.to_datetime(data_df[DATE_COL])
 
date_start = '2023-04-10'
date_end = '2025-11-03'
 
data_df = data_df.loc[data_df[DATE_COL] >= date_start]
 
data_df = data_df.loc[data_df[DATE_COL] <= date_end]
 
data_df.reset_index(drop=True, inplace=True)

data_df[MEDIA_COLS] = data_df[MEDIA_COLS].abs()

data_df.head()


# %% [markdown]
# # External Control Functions

# %% [markdown]
# ## Setting Control Functions

# %% [markdown]
# ### Adding Statsmodel Decomposition

# %%
def create_seasonality_features(data: pd.DataFrame, date_column: str, output_variable: str, monthly_seasonality: int) -> pd.DataFrame:
    """
    Creates seasonality effect features for modeling based on the yearly seasonality using the statsmodels library.

    Parameters:
    - data (pd.DataFrame): The input dataframe.
    - date_column (str): The name of the date column.
    - output_variable (str): The name of the output variable.
    - yearly_seasonality (int): The yearly seasonality period (e.g., 52 for weekly, 365 for daily).

    Returns:
    - pd.DataFrame: The dataframe with added seasonality features.
    """

    data[date_column] = pd.to_datetime(data[date_column])
    
    # Placeholder for results
    results = []
    
    for country in data['Country'].unique():
        # 1. Isolate Geo
        subset = data[data['Country'] == country].copy()
        
        # 2. Sort & Set Index
        subset = subset.sort_values(by=date_column)
        subset.set_index(date_column, inplace=True)
        
        # 3. Decompose (Verify we have enough data points)
        if len(subset) >= (monthly_seasonality * 2):
            decomposition = seasonal_decompose(subset[output_variable], model='additive', period=monthly_seasonality)
            subset[f'{output_variable}_Seasonality'] = decomposition.seasonal.values
        else:
            # Fallback if insufficient data for decomposition
            subset[f'{output_variable}_Seasonality'] = 0
            
        results.append(subset.reset_index())
    
    return pd.concat(results, ignore_index=True)


# %% [markdown]
# ### Adding FRED Data

# %%

def add_fred_data(data: pd.DataFrame, date_column: str, api_key: str, series_map: dict, new_column_name: str) -> pd.DataFrame:
    """
    Fetches data from the FRED API, upsamples it to weekly (Monday), 
    interpolates missing values, and merges it with the MMM dataset.
    """

    """
    Fetches data from the FRED API and merges it with the input dataframe.

    Parameters:
    - data (pd.DataFrame): The input dataframe.
    - date_column (str): The name of the date column in the input dataframe.
    - api_key (str): Your FRED API key.
    - series_id (str): The FRED series ID to fetch.
    - observation_start (str): Start date for observations.
    - observation_end (str): End date for observations.

    Returns:
    - pd.DataFrame: The dataframe with added FRED data.
    """
    
    # 1. Initialize API
    fred = Fred(api_key=api_key)
    data[date_column] = pd.to_datetime(data[date_column])
    
    # Create the new column with NaN
    data[new_column_name] = np.nan
    
    for country, series_id in series_map.items():
        if series_id is None: continue
        
        # 1. Fetch Data
        try:
            print(f"Fetching {series_id} for {country}...")
            # Fetch generic wide range to cover all data
            fred_series = fred.get_series(series_id) 
            fred_df = pd.DataFrame(fred_series, columns=['val'])
            fred_df.index = pd.to_datetime(fred_df.index)
            
            # 2. Resample to Weekly (Monday) & Interpolate
            fred_weekly = fred_df.resample('W-MON').mean().interpolate(method='time')
            fred_weekly = fred_weekly.reset_index().rename(columns={'index': date_column})
            
            # 3. Merge only onto the rows for this country
            # We use a temporary merge to align dates
            subset = data[data['Country'] == country].copy()
            merged = pd.merge(subset, fred_weekly, on=date_column, how='left')
            
            # Fill Gaps
            merged['val'] = merged['val'].fillna(method='ffill').fillna(method='bfill')
            
            # 4. Assign back to main dataframe
            # We use the index to ensure we update the correct rows
            data.loc[data['Country'] == country, new_column_name] = merged['val'].values
            
        except Exception as e:
            print(f"Error fetching {series_id} for {country}: {e}")
            
    return data


# %% [markdown]
# ### Adding Item Trends

# %%
def pytrends_interest_over_time(df: pd.DataFrame, keyword: str, date_col: str) -> pd.DataFrame:
    """
    Fetches Google Trends data using 'trendspy' and fixes the 'list index out of range' error.
    """
    
    # 1. INITIALIZE TRENDSPY
    tr = Trends()
    
    # 2. PREPARE DATAFRAME
    df[date_col] = pd.to_datetime(df[date_col])
    col_name = f'Brand_Trend_Index'
    
    if col_name not in df.columns:
        df[col_name] = 0.0
    
    unique_countries = df['Country'].unique()
    
    for country in unique_countries:
        print(f"Fetching Google Trends for '{keyword}' in {country} via trendspy...")
        
        # Small sleep
        time.sleep(random.uniform(2, 4))
        
        try:
            # --- FETCH DATA ---
            # geo needs to be passed correctly (US, FR, etc.)
            trends = tr.interest_over_time([keyword], timeframe='today 5-y', geo=country)
            
            if trends is not None and not trends.empty:
                # Reset index to turn the 'time [UTC]' index into a column
                trends = trends.reset_index()
                
                # --- FIX FOR "LIST INDEX OUT OF RANGE" ---
                # The column is named 'time [UTC]', so we look for 'time' OR 'date'
                # If neither is found, we default to the first column (index 0)
                date_c = next((c for c in trends.columns if 'date' in c.lower() or 'time' in c.lower()), trends.columns[0])
                
                # The value column is usually the keyword
                val_c = keyword if keyword in trends.columns else trends.columns[1]

                # Rename columns to standard names
                trends = trends.rename(columns={date_c: date_col, val_c: 'val'})
                
                # --- RESAMPLE LOGIC ---
                trends[date_col] = pd.to_datetime(trends[date_col]).dt.tz_localize(None) # Remove UTC timezone if present
                trends = trends.set_index(date_col)
                
                # Resample to Weekly Monday
                trends_weekly = trends[['val']].resample('W-MON').mean().reset_index()

                # --- MERGE LOGIC ---
                mask = df['Country'] == country
                country_subset = df.loc[mask, [date_col]].copy()
                
                merged = pd.merge(country_subset, trends_weekly, on=date_col, how='left')
                merged['val'] = merged['val'].interpolate(method='linear').fillna(0)
                
                # Assign back
                df.loc[mask, col_name] = merged['val'].values
                
                print(f"   -> Success! Added {len(trends_weekly)} data points for {country}.")
            else:
                print(f"   -> Warning: Empty data for {country}")

        except Exception as e:
            print(f"   -> FAILED for {country}: {e}")

    return df


# %% [markdown]
# ### Adding Holiday Control

# %%
def add_holidays(df: pd.DataFrame, date_col: str, country=GEO):
    """
    Given a weekly‐frequency DataFrame `df` (one row per week, with `date_col` as a datetime),
    add one dummy column per federal holiday in `country_code`, where each holiday flags
    the single week whose `date_col` is closest to the official holiday date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a datetime column named `date_col` at weekly frequency (e.g. Mondays).
    date_col : str
        Name of the column in `df` containing the weekly‐period dates.
    country_code : str
        ISO‐3166 alpha-2 code (e.g. "US", "CA", "GB", "DE", "FR", etc.). This is passed
        to `holidays.CountryHoliday` to fetch that country’s official public‐holiday calendar.

    Returns
    -------
    df_out : pd.DataFrame
        A copy of `df` with one new column per holiday.  Each column is 0/1, set to 1
        on the row whose `date_col` is nearest (in absolute days) to that holiday’s official date.
    """

    df[date_col] = pd.to_datetime(df[date_col])
    
    # Initialize column
    df['Holiday_Indicator'] = 0
    
    # Iterate through each country present in the data
    for country_code in df[country].unique():
        
        # Map your dataset codes to 'holidays' library codes
        # 'US' -> 'US', 'FR' -> 'FR'
        # If you have 'UK', map it to 'GB'
        iso_code = 'US' if country_code == 'US' else country_code 
        
        try:
            # 1. Get Years
            subset_mask = df[country] == country_code
            years = df.loc[subset_mask, date_col].dt.year.unique()
            
            # 2. Get Holiday Dates
            country_hols = holidays.CountryHoliday(iso_code, years=years)
            holiday_dates = pd.to_datetime(list(country_hols.keys()))
            
            # 3. Flag Weeks (Vectorized closest match)
            # We look for rows where the date is within 3 days of a holiday
            subset_dates = df.loc[subset_mask, date_col]
            
            for h_date in holiday_dates:
                # Find weeks within ±3 days of the holiday
                mask = (subset_dates - h_date).abs() <= pd.Timedelta(days=3)
                df.loc[subset_mask & mask, 'Holiday_Indicator'] = 1
                
        except NotImplementedError:
            print(f"Warning: Holiday calendar not found for {country_code}")
            
    return df


# %% [markdown]
# ### Adding Anomaly Detection

# %%
# 1. Define the Outlier/Anomaly Calculation Function
def calculate_and_flag_outliers(df, kpi_col=SALES_COL, group_col='Country', multiplier=1.5):
    
    # Dictionary to store limits for verification
    limits_dict = {}
    
    # We will build a list of conditions to apply efficiently
    conditions = []
    
    print(f"{'Geo':<25} | {'Lower Limit ($)':<20} | {'Upper Limit ($)':<20} | {'Max Weekly Rev ($)':<20}")
    print("-" * 95)
    
    # Iterate through every hotel brand to calculate ITS specific limits
    for geo in df[group_col].unique():
        # Get data for this geo
        geo_data = df[df[group_col] == geo][kpi_col]
        
        # Calculate IQR
        Q1 = geo_data.quantile(0.25)
        Q3 = geo_data.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define 1.5x IQR Bounds
        raw_lower_limit = Q1 - (multiplier * IQR)
        lower_limit = max(0, raw_lower_limit)  # <--- FORCE TO 0 IF NEGATIVE
        upper_limit = Q3 + (multiplier * IQR)
        
        # Store for printing
        limits_dict[geo] = (lower_limit, upper_limit)
        
        # Print for User Inspection
        max_val = geo_data.max()
        print(f"{geo:<25} | {lower_limit:,.0f}                | {upper_limit:,.0f}                | {max_val:,.0f}")

    # 2. Apply the Logic (Vectorized)
    # We create a function to apply row-by-row based on the dictionary we just built
    def is_outlier(row):
        geo = row[group_col]
        val = row[kpi_col]
        
        if geo in limits_dict:
            lower, upper = limits_dict[geo]
            # Tag 1 if outside the bounds (High OR Low outlier)
            if val < lower or val > upper:
                return 1
        return 0

    # Apply to create the column
    df['Market_Disruption_Indicator'] = df.apply(is_outlier, axis=1)
    
    return df


# %% [markdown]
# ### Adding Competitors Data Trend

# %%
def add_competitor_composite(df: pd.DataFrame, competitors: list, date_col: str) -> pd.DataFrame:
    """
    Fetches trends for multiple competitors simultaneously using 'trendspy' to preserve relative scale,
    then sums them to create a single 'Competitor_Spend_Index'.
    
    Args:
        competitors: List of keywords (max 5), e.g., ['hp', 'dell', 'asus', 'acer']
    """
    
    # 1. INITIALIZE TRENDSPY
    tr = Trends()
    
    # 2. PREPARE DATAFRAME
    df[date_col] = pd.to_datetime(df[date_col])
    col_name = 'Competitor_Spend_Index'
    
    # Initialize column with 0.0 if it doesn't exist
    if col_name not in df.columns:
        df[col_name] = 0.0
    
    print(f"Generating Composite Index for: {competitors}")
    
    # 3. LOOP THROUGH COUNTRIES
    unique_countries = df['Country'].unique()
    
    for country in unique_countries:
        # Small sleep to avoid rate limits
        sleep_time = random.uniform(3, 6)
        print(f" > {country}: Requesting data... (sleeping {sleep_time:.1f}s)")
        time.sleep(sleep_time)
        
        try:
            # --- FETCH DATA (All competitors in ONE request) ---
            # This ensures relative scaling (e.g. HP=80 vs Dell=40)
            trends = tr.interest_over_time(competitors, timeframe='today 5-y', geo=country)
            
            if trends is not None and not trends.empty:
                # Reset index to turn 'time [UTC]' into a column
                trends = trends.reset_index()
                
                # --- IDENTIFY DATE COLUMN ---
                # Look for 'date' or 'time'
                date_c = next((c for c in trends.columns if 'date' in c.lower() or 'time' in c.lower()), trends.columns[0])
                
                # Rename date column
                trends = trends.rename(columns={date_c: date_col})
                
                # --- RESAMPLE LOGIC ---
                # Remove timezone if present
                trends[date_col] = pd.to_datetime(trends[date_col]).dt.tz_localize(None)
                trends = trends.set_index(date_col)
                
                # Resample strictly the competitor columns to Weekly Monday
                # We select only the columns that match our competitor list
                # (Google sometimes adds other metadata columns)
                valid_cols = [c for c in competitors if c in trends.columns]
                
                if not valid_cols:
                    print(f" > {country}: Warning - Competitor columns missing in response.")
                    continue
                    
                trends_weekly = trends[valid_cols].resample('W-MON').mean()
                
                # --- CREATE COMPOSITE INDEX ---
                # Sum the relative interest of all rivals to get "Total Market Noise"
                trends_weekly['composite_val'] = trends_weekly.sum(axis=1)
                
                # Reset index for merging
                trends_weekly = trends_weekly.reset_index()
                
                # --- MERGE LOGIC ---
                mask = df['Country'] == country
                country_subset = df.loc[mask, [date_col]].copy()
                
                # Merge only the Date and Composite Value
                merged = pd.merge(country_subset, trends_weekly[[date_col, 'composite_val']], on=date_col, how='left')
                
                # Interpolate missing values
                merged['composite_val'] = merged['composite_val'].interpolate(method='linear').fillna(0)
                
                # Assign back to main DataFrame
                df.loc[mask, col_name] = merged['composite_val'].values
                
                print(f" > {country}: Success! Added composite index.")
            else:
                print(f" > {country}: No data returned (Empty response)")
                
        except Exception as e:
            print(f" > {country}: Failed ({e})")
            
    return df


# %% [markdown]
# ### Adding Market Uncertainty

# %%
def add_market_uncertainty_for_sem(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Fetches Market Index data, calculates 'Realized Volatility', 
    and adds it as 'Market_Uncertainty_Index' for SEM analysis.
    """
    ticker_map = {
        'US': '^GSPC', # S&P 500
        'FR': '^FCHI'  # CAC 40
    }
    
    # 1. Ensure Date is datetime and TZ-naive
    df[date_col] = pd.to_datetime(df[date_col])
    if df[date_col].dt.tz is not None:
        df[date_col] = df[date_col].dt.tz_localize(None)

    # 2. Initialize the column if it doesn't exist
    if 'Market_Uncertainty_Index' not in df.columns:
        df['Market_Uncertainty_Index'] = np.nan
    
    print("Generating 'Environmental Uncertainty' (Market Volatility)...")
    
    for country, ticker in ticker_map.items():
        if country not in df['Country'].unique():
            continue
            
        try:
            print(f" > Processing {country} ({ticker})...")
            
            # 3. Fetch Data using Ticker.history (More stable than download)
            stock = yf.Ticker(ticker)
            # Fetch 5y to ensure overlap
            hist = stock.history(period="5y")
            
            if hist.empty:
                print(f" ! Warning: No data returned for {ticker}")
                continue
            
            # 4. Calculate Volatility
            # Calculate daily returns
            hist['Returns'] = hist['Close'].pct_change()
            
            # Calculate rolling 30-day standard deviation (Volatility)
            hist['Volatility'] = hist['Returns'].rolling(window=30).std()
            
            # 5. Resample to Weekly (Monday)
            # Ensure index is datetime
            hist.index = pd.to_datetime(hist.index)
            # CRITICAL: Strip timezone from Yahoo data to match your Excel data
            if hist.index.tz is not None:
                hist.index = hist.index.tz_localize(None)
                
            weekly_vol = hist['Volatility'].resample('W-MON').mean()
            
            # 6. Prepare for Merge
            vol_df = pd.DataFrame(weekly_vol).reset_index()
            vol_df.columns = [date_col, 'New_Volatility_Feature'] # Use temp name to avoid collision
            
            # 7. Merge safely
            # We isolate the country rows
            mask = df['Country'] == country
            subset = df.loc[mask, [date_col]].copy() # Only keep key column to merge
            
            merged = pd.merge(subset, vol_df, on=date_col, how='left')
            
            # Fill gaps (Interpolate)
            merged['New_Volatility_Feature'] = merged['New_Volatility_Feature'].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
            # 8. Assign back to main DataFrame
            # We use the values directly, assuming the sort order hasn't changed (safe with boolean mask)
            df.loc[mask, 'Market_Uncertainty_Index'] = merged['New_Volatility_Feature'].values
            
            print(f" > {country}: Success")
            
        except Exception as e:
            print(f" ! Failed for {country}: {e}")
            
            traceback.print_exc()
            
    return df


# %% [markdown]
# ### Adding Market Volaitility

# %%
def add_trust_proxies(df, spend_cols, sales_col='Revenue', Country = GEO):
    """
    Adds proxies for 'System Reliability' and 'Behavioral Stability' 
    to enable a Trust-based SEM analysis.
    Auto-detects spend columns to ensure Geo-safety.
    """
    df = df.copy()
    
    # --- 1. AUTO-DETECT SPEND COLUMNS ---
    # We grab anything that looks like a Paid Media Spend column.
    # Because France has 0.00 for 'DirectMail', it won't affect the sum.
    # spend_cols = [c for c in df.columns if 'Total_Spend_' in c]
    
    print(f"Calculating Trust Proxies using {len(spend_cols)} spend columns...")
    
    # --- 2. CALCULATE TOTAL WEEKLY PRESSURE ---
    # This sums the money for that specific row's Country.
    df['Total_Spend_Weekly'] = df[spend_cols].sum(axis=1)

    # --- 3. SYSTEM RELIABILITY (Task Difficulty) ---
    # "Is the sales target stable, or does it bounce around wildly?" System Reliability / Task Difficulty
    # We use a Rolling Standard Deviation of Sales.
    # Logic: If sales are bouncing up and down wildly ($1M -> $5M -> $200k), the "System" is unreliable. This creates a difficult environment where panic is likely.
    df['Rolling_Revenue_Volatility'] = df.groupby(Country)[sales_col].transform(
        lambda x: x.rolling(window=4).std()
    ).fillna(0)
    
    # --- 4. BEHAVIORAL STABILITY (Managerial Panic) ---
    # "Did the Manager panic and change the budget drastically?"
    # We look at the % Change in Total Spend. 
    # High Volatility = High Panic = Low Trust.
    
    # Step A: Calculate Week-over-Week % Change
    # We group by Country so US trends don't bleed into France trends
    df['Spend_Pct_Change'] = df.groupby(Country)['Total_Spend_Weekly'].pct_change().fillna(0) # (Spent t - Spend t-1) / Spend t-1
    
    # Step B: Take the Rolling Volatility (Std Dev) of that change -> Behavioral Stability / Managerial Panic
    # Logic: A calm manager increases or decreases budget smoothly. A panicked manager slashes the budget by 50% one week, then doubles it the next. We measure the volatility of the change.
    df['Budget_Allocation_Instability'] = df.groupby(Country)['Spend_Pct_Change'].transform(
        lambda x: x.rolling(window=4).std()
    ).fillna(0)

    # --- 5. SYSTEM SHOCK (Performance Surprise) ---
    # "Did we miss the target last month?" (Disappointment Driver)
    # Formula: (Current Sales - 4wk Avg) / 4wk Avg
    # Logic: Trust is destroyed when expectations are violated. If we expected $10M (based on recent averages) and got $5M, that is a "Negative Shock." This shock drives the Manager to panic in the next step.
    df['Revenue_Variance'] = df.groupby('Country')[sales_col].transform(
        lambda x: (x - x.rolling(window=4).mean()) / x.rolling(window=4).mean()
    ).fillna(0)
    
    # Cleanup intermediate columns if you want (Optional)
    # df.drop(columns=['Spend_Pct_Change'], inplace=True)
        
    return df


# %% [markdown]
# ### Time Series Control

# %%
def add_time_series_controls(df, target = SALES_COL):
    """
    Adds Lagged Revenue to control for Autocorrelation (Inertia).
    Run this BEFORE scaling.
    """
    df = df.copy()
    
    # 1. Sort is CRITICAL (Must be chronological per country)
    df = df.sort_values(by=['Country', 'Week'])
    
    # 2. Create "Inertia" (Lagged Revenue)
    # Group by Country so US sales don't bleed into France rows
    df['Lagged_Revenue'] = df.groupby('Country')[SALES_COL].shift(1)
    
    # 3. Fill NA (The first week of data becomes 0 or the first value)
    # Backfill is safer than 0 so you don't have a massive "drop" at week 1
    df['Lagged_Revenue'] = df.groupby('Country')['Lagged_Revenue'].bfill()
    
    return df


# %% [markdown]
# ## Applying Controls

# %% [markdown]
# ### Seasonality and Trend Effects

# %%
data_df = create_seasonality_features(
    data_df,
    date_column=DATE_COL,
    output_variable=SALES_COL,
    monthly_seasonality=52
)

# %%
data_df.head()

# %% [markdown]
# ### Holidays Effects

# %%
data_df = add_holidays(data_df, DATE_COL)
data_df.head()

# %% [markdown]
# ### Fred Effect

# %%
# 1. Add CPI (Inflation)
# 3. CPI / Inflation
cpi_map = {
    'US': 'CPIAUCSL', 
    'FR': 'FRACPALTT01IXNBM' 
}
data_df = add_fred_data(
    data_df, 
    DATE_COL, 
    api_key='f4de2945a4d9b5d649169d70a58f65d8', 
    series_map=cpi_map, 
    new_column_name='Macro_CPI'
)

data_df.head()

# %%
# 4. Tech Production Index
prod_map = {
    'US': 'IPG334S', 
    'FR': 'FRAPROINDMISMEI'
}
data_df = add_fred_data(
    data_df, 
    DATE_COL, 
    api_key='f4de2945a4d9b5d649169d70a58f65d8', 
    series_map=prod_map, 
    new_column_name='Technological_Production_Index'
)

data_df.head()

# %% [markdown]
# ### Item Trends

# %%
data_df = pytrends_interest_over_time(data_df, keyword='lenovo', date_col=DATE_COL)
data_df.head()
# # data_df['trend_norestate'] = data_df['trend_norestate'].fillna(0)

# %%
data_df = calculate_and_flag_outliers(data_df, kpi_col=SALES_COL, group_col=GEO)

# 3. Verification
data_df.head()


# %% [markdown]
# ### Competitors Effect

# %%
# Define your competitor set (Max 5 keywords allowed by Google per call)
comp_set = ['hp', 'dell', 'asus', 'acer']

# Run the function
data_df = add_competitor_composite(data_df, competitors=comp_set, date_col='Week')

data_df.head()

# %% [markdown]
# ### Market Uncertainty Effects

# %%
data_df = add_market_uncertainty_for_sem(data_df, date_col=DATE_COL)
data_df.head()

# %% [markdown]
# ### Market Volatility Effects

# %%
# Define your spend columns (all columns starting with 'Spend_')
spend_cols = [c for c in data_df.columns if c.startswith('Spend_')]

data_df = add_trust_proxies(data_df, spend_cols, sales_col='Revenue')

data_df.head()

# %% [markdown]
# ### Time Series Control Effects

# %%
data_df = add_time_series_controls(data_df)

data_df.head()

# %% [markdown]
# ## Append Controls & Scale

# %%
CONTROL_COLS = []

# %%
CONTROL_COLS.append(f'{SALES_COL}_Seasonality')
CONTROL_COLS.append('Holiday_Indicator')
CONTROL_COLS.append('Macro_CPI')
CONTROL_COLS.append('Technological_Production_Index')
CONTROL_COLS.append('Brand_Trend_Index')
CONTROL_COLS.append('Market_Disruption_Indicator')
CONTROL_COLS.append('Competitor_Spend_Index')
CONTROL_COLS.append('Market_Uncertainty_Index')
CONTROL_COLS.append('Rolling_Revenue_Volatility')
CONTROL_COLS.append('Budget_Allocation_Instability')
CONTROL_COLS.append('Revenue_Variance')
CONTROL_COLS.append('Lagged_Revenue')

data_df.head()

# %%
data_df.fillna(method='ffill', inplace=True)

# %%
data_df_scld = data_df.copy()
data_df_scld.head()

# %%
# 2. Apply Standard Scaling PER GEO (Crucial for SEM)
print("Standardizing controls per Country (Mean=0, Std=1)...")

NUMERIC_COLS = (
    [SALES_COL] +       # <--- DON'T FORGET THIS!
    MEDIA_COLS +              # Your Paid Media list
    ORGANIC_COLS +            # Your Organic list
    CONTROL_COLS              # Your Controls list
)

NUMERIC_COLS.remove('Holiday_Indicator')
NUMERIC_COLS.remove('Market_Disruption_Indicator')


# for col in CONTROL_COLS:
for col in NUMERIC_COLS:
    # Check if column exists to avoid errors
    if col not in data_df_scld.columns:
        print(f"Skipping {col} (Not found)")
        continue
        
    # Transform strictly within each Country group
    # Logic: (Value - CountryMean) / CountryStd
    data_df_scld[col] = data_df_scld.groupby(GEO)[col].transform(
        lambda x: (x - x.mean()) / x.std()).fillna(0)

# 3. Verification
# The mean for US and FR should both be effectively 0.0
print("\n--- Verification: Means should be ~0.0 ---")
display(data_df_scld.groupby(GEO)[NUMERIC_COLS].mean().round(2))

display(data_df_scld.head())

# %% [markdown]
# ## Check the KPI Trend

# %%
df_lenovo_revenue = data_df.groupby(DATE_COL, as_index=False)[SALES_COL].sum()

fig, ax = plt.subplots(figsize=(16, 4))
fig.patch.set_facecolor("#22272e")
ax.set_facecolor("#22272e")

ax.plot(df_lenovo_revenue[DATE_COL], df_lenovo_revenue[SALES_COL], 
        marker='o', linestyle='-', color='lime', 
        linewidth=2, markersize=6, label="revenue")

ax.set_title("Revenue (y) - All Model Brands", fontsize=14, fontweight='bold', color='white')
ax.set_xlabel("Weeks", fontsize=12, fontweight='bold', color='white', labelpad=8)
ax.set_ylabel("Revenue", fontsize=12, fontweight='bold', color='white', labelpad=8)

ax.grid(True, linestyle='--', alpha=0.4, color='white')
ax.tick_params(axis='x', rotation=45, labelsize=11, colors='white')
ax.tick_params(axis='y', labelsize=11, colors='white')
fig.tight_layout()
plt.show()

# %% [markdown]
# # SEM Analysis

# %% [markdown]
# ### Building the Hypothesis

# %%
'''
Hypothesis Building

Direct Effects (Media Efficiency)
Hypothesis 1: Search Intensity -> Business Performance.
Hypothesis 2: Social Intensity -> Business Performance.Hypothesis 
Hypothesis 3: Brand Intensity -> Business Performance.

Logic: Before assessing trust, we must establish that the marketing investment itself acts as a stimulus for revenue. This aligns with the "Marketing Content" variable in Ferdinand et al. [1], proving that the core input drives the output.
H1: Higher investment in Search Intensity (Performance Marketing) directly increases Sales Revenue.
H2: Higher investment in Social Intensity (Engagement) directly increases Sales Revenue.
H3: Higher investment in Brand Intensity (Awareness) directly increases Sales Reven.

Direct Effects (Antecedents of Trust)
Hypothesis 4: Environmental Uncertainty -> Managerial Adherence.(Note: Negative relationship. Higher Uncertainty reduces Adherence)
Hypothesis 5: System Reliability -> Managerial Adherence.(Note: Positive relationship. Higher Reliability increases Adherence).

Direct Effects (The Cost of Distrust)
Hypothesis 6: Managerial Adherence -> Business Performance.

Indirect (Mediation) Effects 
Hypothesis 7: Environmental Uncertainty -> Managerial Adherence -> Business Performance.

Hypothesis 8: System Reliability -> Managerial Adherence -> Business Performance.

Moderation Effect (Advanced)9.  
Hypothesis 9: Managerial Adherence moderates the relationship: Search Intensity-> Business Performance.(Meaning: High adherence makes Search spend more effective).
'''

# %% [markdown]
# ### Spend Share Analysis

# %%
# ----------------------------
# 1. Spend Share Analysis Global
# ----------------------------
MEDIA_REVENUE = MEDIA_COLS.copy()
# MEDIA_REVENUE+= [SALES_COL]

print("******** Spend Share Analysis Global ********")
if MEDIA_REVENUE:
    # Calculate total spend across all identified media channels.
    total_media_spend = data_df[MEDIA_REVENUE].sum().sum()
    # Compute spend share for each media channel.
    spend_share = data_df[MEDIA_REVENUE].sum() / total_media_spend
    spend_share_df = spend_share.reset_index()
    spend_share_df.columns = ['Media_Channel', 'Spend_Share']
    print(spend_share_df)

    # Plot a bar chart of spend share.
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Spend_Share', y='Media_Channel',
                data=spend_share_df.sort_values(by='Spend_Share', ascending=False))
    plt.title("Spend Share by Media Channel")
    plt.xlabel("Spend Share")
    plt.ylabel("Media Channel")
    plt.show()
else:
    print("No media channels available for spend share analysis.")


# %% [markdown]
# ### VIF Analysis

# %%
def compute_vif(df):
    """Return a DataFrame with VIF for each feature in df."""
    X = df.assign(const=1)  # add intercept
    vif_data = []
    for i, col in enumerate(df.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append((col, vif))
    return pd.DataFrame(vif_data, columns=['feature','VIF'])

print("******** VIF Analysis Paid Media Only ********")

# 1) start with your predictor DataFrame `df`
# df_work = data_df[[*MEDIA_COLS]].dropna()
df_work = data_df_scld[[*MEDIA_COLS]].dropna()


# 2) iterative removal
threshold = 12.0
while True:
    vif_df = compute_vif(df_work)
    max_vif = vif_df['VIF'].max()
    if max_vif <= threshold:
        break

    # drop the worst offending feature
    worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
    print(f"Dropping {worst} (VIF={max_vif:.1f})")
    df_work.drop(columns=[worst], inplace=True)

selected_media = compute_vif(df_work)['feature'].tolist()

print("Final VIFs:\n", compute_vif(df_work))

print("Final VIFs:\n", selected_media)


# %%
def compute_vif(df):
    """Return a DataFrame with VIF for each feature in df."""
    X = df.assign(const=1)  # add intercept
    vif_data = []
    for i, col in enumerate(df.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append((col, vif))
    return pd.DataFrame(vif_data, columns=['feature','VIF'])

print("******** VIF Analysis Paid Media & Controls ********")

# 1) start with your predictor DataFrame `df`
df_work = data_df_scld[[*MEDIA_COLS+CONTROL_COLS]].dropna()
# df_work = data_df[[*MEDIA_COLS]].dropna()


# 2) iterative removal
threshold = 12.0
while True:
    vif_df = compute_vif(df_work)
    max_vif = vif_df['VIF'].max()
    if max_vif <= threshold:
        break

    # drop the worst offending feature
    worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
    print(f"Dropping {worst} (VIF={max_vif:.1f})")
    df_work.drop(columns=[worst], inplace=True)

selected_media = compute_vif(df_work)['feature'].tolist()

print("Final VIFs:\n", compute_vif(df_work))

print("Final VIFs:\n", selected_media)


# %%
def compute_vif(df):
    """Return a DataFrame with VIF for each feature in df."""
    X = df.assign(const=1)  # add intercept
    vif_data = []
    for i, col in enumerate(df.columns):
        vif = variance_inflation_factor(X.values, i)
        vif_data.append((col, vif))
    return pd.DataFrame(vif_data, columns=['feature','VIF'])

print("******** VIF Analysis Per Geo Paid Media, Organic, and Controls ********")


for i in data_df_scld[GEO].unique():
    print(f"==== Geo {i} ===")
    # 1) start with your predictor DataFrame `df`
    contribution_df = data_df_scld.loc[data_df_scld[GEO] == i]
    df_work = contribution_df[[*MEDIA_COLS+CONTROL_COLS+ORGANIC_COLS]].dropna()

    # 2) iterative removal
    threshold = 10.0
    while True:
        vif_df = compute_vif(df_work)
        max_vif = vif_df['VIF'].max()
        if max_vif <= threshold:
            break

        # drop the worst offending feature
        worst = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        print(f"Dropping {worst} (VIF={max_vif:.1f})")
        df_work.drop(columns=[worst], inplace=True)

    selected_media = compute_vif(df_work)['feature'].tolist()

    print("Final VIFs:\n", compute_vif(df_work))

    print("Final VIFs:\n", selected_media)

# %%
data_df_scld.drop(['Technological_Production_Index'], axis=1, inplace=True)

items_to_remove = {'Technological_Production_Index'}
CONTROL_COLS = [col for col in CONTROL_COLS if col not in items_to_remove]

# %% [markdown]
# ### Correlation Heatmap Analysis

# %%
# plt.figure(figsize=(20, 10)) 
# dataplot = sns.heatmap(data_df[[*MEDIA_COLS]].corr(numeric_only=True),  annot=True)
# plt.show()

plt.figure(figsize=(20, 10)) 
dataplot = sns.heatmap(data_df_scld[[*MEDIA_COLS+CONTROL_COLS]].corr(numeric_only=True),  annot=True)
plt.show()



# %%
data_df_scld[[*MEDIA_COLS+CONTROL_COLS]].corr(numeric_only=True)

# %% [markdown]
# ## SEM Model

# %%
print("Media Cols:", len(MEDIA_COLS))
print("Control Cols:", len(CONTROL_COLS))

# %%
MEDIA_COLS

# %%
CONTROL_COLS

# %%
ORGANIC_COLS

# %% [markdown]
# ### Experiment Analysis 1

# %%
from semopy import Model

# ==============================================================================
# 1. PREPARE THE US DATASET
# ==============================================================================
print("Filtering for US Market...")
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# 2. DEFINE THE STRUCTURAL EQUATION MODEL (US ONLY)
# ==============================================================================

# A. The "Trust Chain" (Updated with Holiday Control)
trust_mechanism = """
    # 1. THE TRIGGER: What causes Spend Volatility?
    # We add 'is_any_fed_holiday_week' here to filter out "Strategic Volatility" (like Black Friday).
    # The remaining volatility is "Panic" driven by Uncertainty and Sales Shocks.
    Budget_Allocation_Instability ~ Market_Uncertainty_Index + Competitor_Spend_Index + Rolling_Revenue_Volatility + Revenue_Variance + Holiday_Indicator
    
    # 2. THE CONSEQUENCE: Panic destroys Revenue
    Revenue ~ Budget_Allocation_Instability
"""

# B. The "Media Mix" (Added Organic Channels)
# Total_Spend_Search_Brand: (Defensive). The user types "Lenovo Laptop". They already know you. You pay for this ad just to make sure HP or Dell doesn't steal the spot at the top of the page.
# Total_Spend_Search_NonBrand: (Offensive). The user types "Best gaming laptop 2025". They don't have a specific brand in mind. You pay to show up here to introduce them to Lenovo.
# Total_Spend_Search_Shopping: (The Digital Shelf). The user sees a picture of the laptop with a price tag right on the Google results page. This is high-intent "Window Shopping."
# Total_Spend_Social_Meta: (Facebook & Instagram). The biggest social network. Good for broad awareness and retargeting people who visited your site.
# Total_Spend_Social_LinkedIn: (The Professionals). Targeting IT Managers and CTOs. "Buy Lenovo for your whole office.
# Total_Spend_Social_TikTok: (The Discovery Engine). Short, viral videos. Usually younger audience, but great for "This laptop looks cool" vibes.
# Total_Spend_Social_Reddit: (The Communities). Targeting specific subreddits like r/gaming or r/laptops. Very high trust, but hard to get right.
# Total_Spend_Social_Snapchat / Pinterest: Niche visual platforms. Pinterest is often for planning (e.g., "Home Office Setup").
# Total_Spend_Video_YouTube: (Pre-Roll). The "Skip in 5 seconds" ads. It acts like a search engine but for video reviews.
# Total_Spend_Video_CTV: (Connected TV). This is the TV in the living room (Roku, AppleTV, Hulu). It looks like a TV commercial, but you buy it digitally.
# Total_Spend_Video_OLV: (Online Video). Video ads that appear on news sites (CNN, ESPN) or blogs. These are not YouTube; they are videos on the "Open Web."
# Total_Spend_Display_Programmatic: (Banner Ads). These are the image rectangles that follow you around the internet after you look at a pair of shoes. It keeps Lenovo top-of-mind.
# Total_Spend_Audio: (Ears Only). Spotify ads, Pandora, or Podcast sponsorships. Good for telling a story when people can't look at a screen.
# Total_Spend_Affiliate: (Commission). You pay sites like "RetailMeNot" or "TechRadar" a commission if they send a customer who buys something. You only pay when you sell.
# Total_Spend_DirectMail: (Snail Mail). Physical postcards or catalogs sent to people's mailboxes.

media_mix = """
    # --- Direct Effects (Sales Capture / Closers) ---
    # Organic Channels (New!)
    Revenue ~ Organic_Email + Organic_Social

    # Paid Channels
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic + Spend_Shopping
    Revenue ~ Spend_Meta + Spend_LinkedIn + Spend_TikTok
    Revenue ~ Spend_Snapchat + Spend_Reddit + Spend_Pinterest
    Revenue ~ Spend_YouTube + Spend_CTV + Spend_OLV
    Revenue ~ Spend_Display + Spend_Audio
    Revenue ~ Spend_Affiliate + Spend_Direct_Mail

    # --- Indirect Effects (The Upper Funnel Assist) ---
    # Organic Social drives Brand Search just like Paid Social
    Spend_Search_Brand ~ Organic_Social
    
    # Paid Assists
    Spend_Search_Brand ~ Spend_TikTok + Spend_Meta + Spend_YouTube
    Spend_Search_Generic ~ Spend_OLV + Spend_Display
"""

# C. The "Baseline Controls" (Revenue Drivers)
# We keep Seasonality here to explain natural revenue waves.
# We dropped 'Control_Tech_Production' to fix the VIF issue.
base_controls = """
    Revenue ~ Revenue_Seasonality + Brand_Trend_Index + Market_Disruption_Indicator + Lagged_Revenue + Holiday_Indicator
"""

# Combine everything
model_us_desc = trust_mechanism + media_mix + base_controls

print(model_us_desc)

# %%

# ==============================================================================
# 3. RUN THE US SEM
# ==============================================================================
print("\n========== 1. RUNNING US STRUCTURAL EQUATION MODEL ==========")
m_us = Model(model_us_desc)
res_us = m_us.fit(us_data) 
ins_us = m_us.inspect()
print("US SEM Converged Successfully.")

# Print out the Significant Paths to check our Hypotheses
print("\n--- SIGNIFICANT PATHS (p < 0.10) ---")
sig_paths = ins_us[ins_us['p-value'] < 0.10].sort_values(by='Estimate', ascending=False)
print(sig_paths[['lval', 'op', 'rval', 'Estimate', 'p-value']].to_string(index=False))

display(ins_us)

# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_us, "US_SEM_Path_Diagram.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='1') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
from IPython.display import Image, display
display(Image("US_SEM_Path_Diagram.png"))

print("Diagram saved as 'US_SEM_Path_Diagram.png'")

# %% [markdown]
# ### Experiment Analysis 2

# %%
# ==============================================================================
# TEST: REMOVING COMPETITOR INDEX TO FIX MULTICOLLINEARITY
# ==============================================================================

# A. The "Trust Chain" (Removed Competitor Index)
trust_mechanism_v2 = """
    Budget_Allocation_Instability ~ Market_Uncertainty_Index + Rolling_Revenue_Volatility + Revenue_Variance + Holiday_Indicator
    Revenue ~ Budget_Allocation_Instability
"""

# B. The "Media Mix" (Unchanged)
media_mix_v2 = """
    Revenue ~ Organic_Email + Organic_Social
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic + Spend_Shopping
    Revenue ~ Spend_Meta + Spend_LinkedIn + Spend_TikTok
    Revenue ~ Spend_Snapchat + Spend_Reddit + Spend_Pinterest
    Revenue ~ Spend_YouTube + Spend_CTV + Spend_OLV
    Revenue ~ Spend_Display + Spend_Audio
    Revenue ~ Spend_Affiliate + Spend_Direct_Mail
    
    Spend_Search_Brand ~ Organic_Social
    Spend_Search_Brand ~ Spend_TikTok + Spend_Meta + Spend_YouTube
    Spend_Search_Generic ~ Spend_OLV + Spend_Display
"""

# C. The "Baseline Controls" (Removed Competitor Index here too just in case)
base_controls_v2 = """
    Revenue ~ Revenue_Seasonality + Brand_Trend_Index + Market_Disruption_Indicator + Lagged_Revenue + Holiday_Indicator
"""

model_v2_desc = trust_mechanism_v2 + media_mix_v2 + base_controls_v2

print("========== RUNNING V2: NO COMPETITOR INDEX ==========")
m_v2 = Model(model_v2_desc)
res_v2 = m_v2.fit(us_data) 
ins_v2 = m_v2.inspect()

display(ins_v2)

# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_v2, "US_SEM_Path_Diagram_V2.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='1') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_V2', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
display(Image("US_SEM_Path_Diagram_V2.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_V2.png'")

# %% [markdown]
# ### Experiment Analysis 3

# %%

# ==============================================================================
# 1. PREPARE THE US DATASET
# ==============================================================================
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# 2. DEFINE THE REFINED MODEL (The "Clean" Version)
# ==============================================================================

# A. The Trust Chain (REMOVED: Competitor Index, Performance Surprise)
# We focus ONLY on the things that actually worked: Volatility & Uncertainty.



trust_mechanism_v3 = """
    Budget_Allocation_Instability ~ Market_Uncertainty_Index + Rolling_Revenue_Volatility + Holiday_Indicator
    Revenue ~ Budget_Allocation_Instability
"""

# B. The Media Mix (Unchanged for now, let's see if they wake up)
media_mix_v3 = """
    # --- Direct Effects ---
    Revenue ~ Organic_Email + Organic_Social
    
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic + Spend_Shopping
    Revenue ~ Spend_Meta + Spend_LinkedIn + Spend_TikTok
    Revenue ~ Spend_Snapchat + Spend_Reddit + Spend_Pinterest
    Revenue ~ Spend_YouTube + Spend_CTV + Spend_OLV
    Revenue ~ Spend_Display + Spend_Audio
    Revenue ~ Spend_Affiliate + Spend_Direct_Mail

    # --- Indirect Effects (Assists) ---
    Spend_Search_Brand ~ Organic_Social
    Spend_Search_Brand ~ Spend_TikTok + Spend_Meta + Spend_YouTube
    Spend_Search_Generic ~ Spend_OLV + Spend_Display
"""

# C. Baseline Controls (REMOVED: Trend Lenovo)
# We rely on Seasonality + CPI to explain "Time".
base_controls_v3 = """
    Revenue ~ Revenue_Seasonality +  Market_Disruption_Indicator + Lagged_Revenue + Holiday_Indicator + Macro_CPI
"""

# Combine
model_v3_desc = trust_mechanism_v3 + media_mix_v3 + base_controls_v3

# ==============================================================================
# 3. RUN THE PRUNED SEM
# ==============================================================================
print("\n========== RUNNING V3: PRUNED MODEL ==========")
m_v3 = Model(model_v3_desc)
res_v3 = m_v3.fit(us_data) 
ins_v3 = m_v3.inspect()

print("V3 Converged.\n")

# Print Significant Paths
print("--- SIGNIFICANT PATHS (p < 0.10) ---")
sig_paths_v3 = ins_v3[ins_v3['p-value'] < 0.10].sort_values(by='Estimate', ascending=False)
print(sig_paths_v3[['lval', 'op', 'rval', 'Estimate', 'p-value']].to_string(index=False))

display(ins_v3)

# %%

# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_v3, "US_SEM_Path_Diagram_V3.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='1') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_V3', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
display(Image("US_SEM_Path_Diagram_V3.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_V3.png'")

# %% [markdown]
# ### Experiment Analysis 4

# %%
# ==============================================================================
# 1. PREPARE THE US DATASET
# ==============================================================================
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# 2. DEFINE THE REFINED MODEL (The "Clean" Version)
# ==============================================================================

# A. The Trust Chain (REMOVED: Competitor Index, Performance Surprise)
# We focus ONLY on the things that actually worked: Volatility & Uncertainty.
# Updated Trust Mechanism
trust_mechanism_v4 = """
    Budget_Allocation_Instability ~ Market_Uncertainty_Index + Rolling_Revenue_Volatility + Holiday_Indicator
    Revenue ~ Budget_Allocation_Instability
"""

# B. Updated Media Mix (Direct and Indirect/Assists)
media_mix_v4 = """
    # --- Direct Effects ---
    Revenue ~ Organic_Email + Organic_Social
    
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic + Spend_Shopping
    Revenue ~ Spend_Meta + Spend_LinkedIn + Spend_TikTok
    Revenue ~ Spend_Snapchat + Spend_Reddit + Spend_Pinterest
    Revenue ~ Spend_YouTube + Spend_CTV + Spend_OLV
    Revenue ~ Spend_Display + Spend_Audio
    Revenue ~ Spend_Affiliate + Spend_Direct_Mail

    # --- Indirect Effects (Assists) ---
    Spend_Search_Brand ~ Organic_Social
    Spend_Search_Brand ~ Spend_TikTok + Spend_Meta + Spend_YouTube
    Spend_Search_Brand ~ Spend_LinkedIn + Spend_Snapchat + Spend_Reddit + Spend_Pinterest
    
    Spend_Search_Generic ~ Spend_OLV + Spend_Display
    Spend_Search_Generic ~ Organic_Social
    Spend_Search_Generic ~ Spend_TikTok + Spend_Meta + Spend_YouTube
    Spend_Search_Generic ~ Spend_LinkedIn + Spend_Snapchat + Spend_Reddit + Spend_Pinterest
"""

# C. Updated Baseline Controls
base_controls_v4 = """
    Revenue ~ Revenue_Seasonality + Market_Uncertainty_Index + Market_Disruption_Indicator + Lagged_Revenue + Holiday_Indicator
"""

# Combine
model_v4_desc = trust_mechanism_v4 + media_mix_v4 + base_controls_v4

# ==============================================================================
# 3. RUN THE PRUNED SEM
# ==============================================================================
print("\n========== RUNNING v4: PRUNED MODEL ==========")
m_v4 = Model(model_v4_desc)
res_v4 = m_v4.fit(us_data) 
ins_v4 = m_v4.inspect()

print("v4 Converged.\n")

# Print Significant Paths
print("--- SIGNIFICANT PATHS (p < 0.10) ---")
sig_paths_v4 = ins_v4[ins_v4['p-value'] < 0.10].sort_values(by='Estimate', ascending=False)
print(sig_paths_v4[['lval', 'op', 'rval', 'Estimate', 'p-value']].to_string(index=False))

# Comparisons
print("\n--- CHECK: Did 'Behavior_Spend_Volatility' get stronger? ---")
print(ins_v4[ins_v4['rval'] == 'Behavior_Spend_Volatility'][['lval', 'op', 'rval', 'Estimate', 'p-value']].to_string(index=False))

display(ins_v4)

# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_v4, "US_SEM_Path_Diagram_v4.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='1') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_v4', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
display(Image("US_SEM_Path_Diagram_v4.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_v4.png'")

# %%
len(MEDIA_COLS)

# %% [markdown]
# ### Experiment Analysis 5

# %%
# ==============================================================================
# 1. PREPARE THE US DATASET
# ==============================================================================
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# 2. DEFINE THE REFINED MODEL (The "Clean" Version)
# ==============================================================================

# A. The Trust Chain (REMOVED: Competitor Index, Performance Surprise)
# We focus ONLY on the things that actually worked: Volatility & Uncertainty.
# Updated Trust Mechanism
trust_mechanism_v5 = """
    Budget_Allocation_Instability ~ Rolling_Revenue_Volatility 
    Budget_Allocation_Instability ~ Market_Uncertainty_Index + Market_Disruption_Indicator
"""

# B. Updated Media Mix (Direct and Indirect/Assists)
media_mix_v5 = """
    # --- Direct Effects ---
    Revenue ~ Organic_Email + Organic_Social 
    Revenue ~ Budget_Allocation_Instability
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic 
    Revenue ~ Holiday_Indicator 
    Revenue ~ Spend_Reddit  
    Revenue ~ Spend_Direct_Mail
    Revenue ~ Competitor_Spend_Index  

    # --- Indirect Effects (Assists) ---
    Spend_Search_Brand ~ Spend_Meta   
    Spend_Search_Brand ~ Spend_Shopping + Spend_OLV 
    
    Spend_Search_Generic ~ Spend_YouTube + Spend_Display  
    Spend_Search_Generic ~ Spend_Snapchat + Spend_Shopping + Spend_OLV + Spend_CTV
    
    Spend_CTV ~ Spend_Audio
    Spend_OLV ~ Spend_Audio
    
    Spend_YouTube ~ Spend_TikTok
    
    Spend_Meta ~ Spend_Pinterest + Spend_LinkedIn + Spend_Affiliate
"""

# C. Updated Baseline Controls
base_controls_v5 = """
    Revenue ~ Revenue_Seasonality + Macro_CPI   
"""

# Combine
model_v5_desc = trust_mechanism_v5 + media_mix_v5 + base_controls_v5

# ==============================================================================
# 3. RUN THE PRUNED SEM
# ==============================================================================
print("\n========== RUNNING v5: PRUNED MODEL ==========")
m_v5 = Model(model_v5_desc)
res_v5 = m_v5.fit(us_data) 
ins_v5 = m_v5.inspect()

print("v5 Converged.\n")

# Print Significant Paths
print("--- SIGNIFICANT PATHS (p < 0.10) ---")
sig_paths_v5 = ins_v5[ins_v5['p-value'] < 0.10].sort_values(by='Estimate', ascending=False)
print(sig_paths_v5[['lval', 'op', 'rval', 'Estimate', 'p-value']].to_string(index=False))

display(ins_v5)

# Comparisons 
print("========== Number of Accepted P Values ==========")
print(len(sig_paths_v5))


# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_v5, "US_SEM_Path_Diagram_v5.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='0.5') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_v5', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
display(Image("US_SEM_Path_Diagram_v5.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_v5.png'")

# %% [markdown]
# ### Experiment Analysis 6

# %%
# ==============================================================================
# DEFINE MODEL V6: THE "TRUST" MODEL (Causality vs. Correlation)
# ==============================================================================
# ==============================================================================
# 1. PREPARE THE US DATASET
# ==============================================================================
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# 2. DEFINE THE REFINED MODEL (The "Clean" Version)
# ==============================================================================

# A. The Trust Chain (REMOVED: Competitor Index, Performance Surprise)
# We focus ONLY on the things that actually worked: Volatility & Uncertainty.

# The Trust Chain (Human Behavior = CAUSAL)
# Managers reacting to volatility is a causal behavior.


# A. The Trust Chain
# Updated Trust Mechanism
trust_mechanism_v6 = """
    Budget_Allocation_Instability ~ Rolling_Revenue_Volatility + Market_Uncertainty_Index + Market_Disruption_Indicator
    Revenue ~ Budget_Allocation_Instability
"""

# B. Updated Media Mix (Direct Effects & Assists)
media_mix_v6 = """
    # --- Direct Effects (The Closers) ---
    Revenue ~ Organic_Email + Organic_Social
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic 
    Revenue ~ Spend_Reddit + Spend_Direct_Mail
    Revenue ~ Holiday_Indicator + Competitor_Spend_Index
    
    # --- The "Orphans" (Included for Matrix Completeness) ---
    Revenue ~ Spend_LinkedIn + Spend_Pinterest + Spend_Affiliate + Spend_TikTok
    Revenue ~ Spend_CTV + Spend_OLV + Spend_YouTube + Spend_Audio

    # --- Indirect Effects (The Assists) ---
    Spend_Search_Brand    ~ Spend_Meta + Spend_YouTube + Spend_Shopping
    Spend_Search_Generic ~ Spend_Display + Spend_YouTube + Spend_Shopping
    
    # --- The "Shopping" Effect ---
    Spend_Search_Brand    ~ Spend_Shopping
    Spend_Search_Generic ~ Spend_Shopping
"""

# C. Updated Budget Correlations (Managerial Strategy)
budget_correlations_v6 = """
    # The "Social Bundle"
    Spend_Meta ~~ Spend_LinkedIn
    Spend_Meta ~~ Spend_Pinterest
    Spend_Meta ~~ Spend_Affiliate
    Spend_Meta ~~ Spend_TikTok
    
    # The "Video Bundle"
    Spend_CTV ~~ Spend_Audio
    Spend_OLV ~~ Spend_Audio
    Spend_YouTube ~~ Spend_TikTok
"""

# D. Updated Baseline Controls
base_controls_v6 = """
    Revenue ~ Revenue_Seasonality + Macro_CPI
"""

# Combine
model_v6_desc = trust_mechanism_v6 + media_mix_v6 + budget_correlations_v6 + base_controls_v6

# Run the Model
print("Running V6 (Trust Model - Fixed)...")
m_v6 = semopy.Model(model_v6_desc)
res_v6 = m_v6.fit(us_data)
ins_v6 = m_v6.inspect()
print("Success! V6 Converged.")

display(ins_v6)

# %%
sig_paths_v6 = ins_v6[ins_v6['p-value'] < 0.10].sort_values(by='Estimate', ascending=False)
sig_paths_v6

# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_v6, "US_SEM_Path_Diagram_v6.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR') 

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square. 
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='0.5') 

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
g.attr(nodesep='0.1', ranksep='1.0')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_v6', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
display(Image("US_SEM_Path_Diagram_v6.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_v6.png'")

# %% [markdown]
# ### Final Result

# %%
# ==============================================================================
# 0. PREPARE THE US DATASET
# ==============================================================================
print("Filtering for US Market...")
us_data = data_df_scld[data_df_scld['Country'] == 'US'].copy()

# ==============================================================================
# DEFINE MODEL V8: THE "FINAL VALIDATED" MODEL
# Based on your provided P-value table
# ==============================================================================

# 1. Managerial Decision Dynamics (The "Panic" Mechanism)
# Results confirm: Sales Volatility (0.44) and Market Uncertainty (0.15) drive Spend Volatility.
decision_dynamics = """
    Budget_Allocation_Instability ~ Rolling_Revenue_Volatility + Market_Uncertainty_Index + Market_Disruption_Indicator
"""

# 2. Revenue Drivers (The "Closers")
# Results confirm: Reddit (0.18), LinkedIn (0.10), and Brand Search (0.21) are the main drivers.
revenue_drivers = """
    Revenue ~ Spend_Search_Brand + Spend_Search_Generic
    Revenue ~ Spend_Reddit + Spend_LinkedIn
    Revenue ~ Competitor_Spend_Index + Holiday_Indicator
    Revenue ~ Budget_Allocation_Instability
"""

# 3. Funnel Mechanics (The "Assists")
# Results confirm: Meta (0.73) and Shopping (0.35) drive Brand Search.
# Results confirm: Display (0.46) drives Non-Brand Search.
funnel_mechanics = """
    # --- Brand Search Drivers ---
    Spend_Search_Brand ~ Spend_Meta + Spend_Shopping + Spend_YouTube
    
    # --- Non-Brand Search Drivers ---
    Spend_Search_Generic ~ Spend_Display + Spend_Shopping
"""

# 4. Budget Correlations (The "Trust" Fix)
# These explain the shared variance between channels.
budget_correlations = """
    # Social Bundle
    Spend_Meta ~~ Spend_TikTok
    Spend_Meta ~~ Spend_LinkedIn
    Spend_Meta ~~ Spend_Affiliate
    Spend_Meta ~~ Spend_Pinterest
    
    # Video/Audio Bundle
    Spend_YouTube ~~ Spend_TikTok
    Spend_CTV     ~~ Spend_Audio
    Spend_OLV     ~~ Spend_Audio
"""

# 5. Macro Controls & Technical Fixes
# We MUST include TikTok, Audio, etc. in the regression to prevent "Not in List" errors,
# even if they are not significant drivers of revenue.
controls = """
    Revenue ~ Revenue_Seasonality + Macro_CPI
    
    # Technical Controls (Required for Covariance to work)
    Revenue ~ Spend_TikTok + Spend_Affiliate + Spend_Pinterest
    Revenue ~ Spend_YouTube + Spend_CTV + Spend_OLV + Spend_Audio
"""

# Combine all parts
model_final_desc = decision_dynamics + revenue_drivers + funnel_mechanics + budget_correlations + controls

# ==============================================================================
# RUN THE MODEL
# ==============================================================================
print("Running Final Validated Model...")
m_final = semopy.Model(model_final_desc)
res_final = m_final.fit(us_data) 
ins_final = m_final.inspect()

# Show only the significant paths from your table
print("\n--- CONFIRMED SIGNIFICANT PATHS ---")
print(ins_final[ins_final['p-value'] < 0.10].sort_values(by='p-value'))

# ==============================================================================
# GENERATE THE PATH DIAGRAM
# ==============================================================================
print("\nGenerating Diagram...")
# Ensure Graphviz is in PATH
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

g = semopy.semplot(m_final, "Final_Path_Diagram.png", plot_covs=True, std_ests=True)

# Formatting to make it wide and readable
g.attr(rankdir='LR') 
g.attr(ratio='fill') 
g.attr(size='20,12') # Wide canvas
g.attr(nodesep='0.5') 
g.attr(overlap='false')
g.attr(splines='true')

g.render('Final_Path_Diagram', format='png', cleanup=True)
display(Image("Final_Path_Diagram.png"))

# %%
g = semopy.semplot(m_final, "Final_Path_Diagram.png", plot_covs=True, std_ests=True)

# Formatting to make it wide and readable
g.attr(rankdir='LR') 
g.attr(ratio='fill') 
g.attr(size='20,12') # Wide canvas
g.attr(nodesep='0') 
g.attr(overlap='false')
g.attr(splines='true')

g.render('Final_Path_Diagram', format='png', cleanup=True)
display(Image("Final_Path_Diagram.png"))

# %%
ins_final

# %%
stats = semopy.calc_stats(m_final)
stats

# %%
# 1. CREATE BLIND TEST DATA (Remove the answer!)
# We drop 'Revenue' so the model is FORCED to calculate it using only Ad Spend.

# 2. PREDICT
print("Generating Blind Predictions...")

# ==========================================================
# 1. GENERATE PREDICTIONS
# ==========================================================
# semopy calculates the value of EVERY variable based on its parents.
# It uses the path coefficients (betas) you just found.
X_blind = us_data.drop(columns=['Revenue'])

print("Generating Causal Predictions...")
blind_preds = m_final.predict(X_blind)

# ==========================================================
# 2. VALIDATE THE MODEL (The Proof)
# ==========================================================
actual = us_data['Revenue']
predicted = blind_preds['Revenue']

# Calculate the TRUE R-Squared (Should match your 0.82 result)
r2 = r2_score(actual, predicted)
mae = mean_absolute_error(actual, predicted)

print(f"\n--- FINAL PREDICTIVE ACCURACY ---")
print(f"R-Squared: {r2:.4f} (This verifies your 82% fit)")
print(f"Mean Error: {mae:.4f} (In scaled units)")

# ==========================================================
# 3. VISUALIZE THE FIT
# ==========================================================
plt.figure(figsize=(12, 6))

# Plot Actual vs Predicted
plt.plot(actual.values, label='Actual Revenue', color='black', alpha=0.6)
plt.plot(predicted.values, label='SEM Predicted Revenue', color='blue', linewidth=2)

plt.title(f"SEM Prediction: Actual vs. Model (R² = {r2:.2f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# ==========================================================
# 4. THE "KILLER FEATURE": INTERMEDIATE PREDICTION
# ==========================================================
# Prove that you aren't just guessing Revenue, but explaining the CAUSE.
# Let's see if the model correctly predicted 'Brand Search' volume too.
# plt.figure(figsize=(12, 6))
# plt.plot(us_data['Spend_Search_Brand'].values, label='Actual Brand Search', color='gray', alpha=0.5)
# plt.plot(blind_preds['Spend_Search_Brand'].values, label='Predicted Brand Search (Driven by Meta)', color='green', linewidth=2)
# plt.title("Mechanism Check: Did we correctly predict Brand Search volume?")
# plt.legend()
# plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=actual, y=predicted, alpha=0.6)
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2) # The Perfect Line
plt.xlabel("Actual Revenue (Standardized)")
plt.ylabel("SEM Predicted Revenue")
plt.title(f"True Model Fit (R² = {r2:.2f})")
plt.show()

# %%



# 1. Get the Original Stats (The "Key" to un-scale)
# Ensure 'data_df' is your ORIGINAL un-scaled dataframe
mu = data_df['Revenue'].mean()
sigma = data_df['Revenue'].std()

# 2. Get the Scaled Values from your SEM Blind Prediction
# 'actual' is the Z-score from us_data
# 'predicted' is the Z-score from blind_preds
y_scaled = actual 
y_pred_scaled = predicted

# 3. Un-Scale Back to Dollars (The "Unlock")
y_actual_dollars = (y_scaled * sigma) + mu
y_pred_dollars = (y_pred_scaled * sigma) + mu

# 4. Calculate MAPE in Dollars
mape = mean_absolute_percentage_error(y_actual_dollars, y_pred_dollars)
rmse_value = root_mean_squared_error(y_actual_dollars, y_pred_dollars)

print(f"--- FINAL SEM PERFORMANCE ---")
print(f"R-Squared (Scientific Fit): {0.8177:.4f}")
print(f"MAPE (Business Accuracy):   {mape:.2%}")
print(f"RMSE: {rmse_value}")


# %%
# ==========================================================
# 5-ITERATION 70:30 CROSS-VALIDATION
# ==========================================================
print("\nStarting 10-Iteration 80:20 Cross-Validation...")

# Initialize ShuffleSplit: 5 iterations, 30% test size (70% train)
cv = ShuffleSplit(n_splits=10, test_size=0.20, random_state=42)

cv_r2_scores = []
cv_mae_scores = []

# Loop through the 5 different train/test splits
for fold, (train_idx, test_idx) in enumerate(cv.split(us_data), 1):
    
    # 1. Create the 70% Train and 30% Test DataFrames
    train_df = us_data.iloc[train_idx].copy()
    test_df = us_data.iloc[test_idx].copy()
    
    # 2. Initialize a fresh model for this fold to avoid data leakage
    m_cv = semopy.Model(model_final_desc)
    
    try:
        # 3. Fit the model strictly on the 70% TRAIN data
        m_cv.fit(train_df)
        
        # 4. Create blind TEST data (Drop 'Revenue')
        X_test_blind = test_df.drop(columns=['Revenue'])
        
        # 5. Predict on the blind TEST data
        blind_preds_cv = m_cv.predict(X_test_blind)
        
        # 6. Evaluate Accuracy
        actual_cv = test_df['Revenue']
        predicted_cv = blind_preds_cv['Revenue']
        
        fold_r2 = r2_score(actual_cv, predicted_cv)
        fold_mae = mean_absolute_error(actual_cv, predicted_cv)
        
        cv_r2_scores.append(fold_r2)
        cv_mae_scores.append(fold_mae)
        
        print(f"Fold {fold}: Out-of-Sample R² = {fold_r2:.4f}, MAE = {fold_mae:.4f}")
        
    except Exception as e:
        # SEM models sometimes fail to converge if a specific random 70% slice 
        # lacks enough variance in certain variables.
        print(f"Fold {fold} encountered a convergence or fitting error: {e}")

# ==========================================================
# FINAL CROSS-VALIDATION RESULTS
# ==========================================================
print("\n--- FINAL OUT-OF-SAMPLE PREDICTIVE ACCURACY (80:20 SPLIT) ---")
if cv_r2_scores:
    print(f"Average R-Squared: {np.mean(cv_r2_scores):.4f} ± {np.std(cv_r2_scores):.4f}")
    print(f"Average Mean Error: {np.mean(cv_mae_scores):.4f} ± {np.std(cv_mae_scores):.4f}")
else:
    print("No folds completed successfully. Check your data variance.")

# %%
print("\nStarting Anchored Time Series Cross-Validation...")

# We need to ensure chronological order first!
# us_data = us_data.sort_values('Date_Column') # Uncomment if you have a date column

total_rows = len(us_data)
# Force the model to ALWAYS train on at least 50% of the data to prevent "Not PD" errors
initial_train_size = int(total_rows * 0.50) 
test_window_size = int(total_rows * 0.10) # Test on 10% chunks rolling forward

# Calculate how many folds we can safely do with this math
n_splits = int((total_rows - initial_train_size) / test_window_size)

ts_r2_scores = []
ts_mae_scores = []

for fold in range(n_splits):
    # Expanding the training window: 50% -> 60% -> 70% -> 80%
    train_end = initial_train_size + (fold * test_window_size)
    test_end = train_end + test_window_size
    
    train_df = us_data.iloc[:train_end].copy()
    test_df = us_data.iloc[train_end:test_end].copy()
    
    m_ts = semopy.Model(model_final_desc)
    
    try:
        m_ts.fit(train_df)
        
        X_test_blind = test_df.drop(columns=['Revenue'])
        blind_preds_ts = m_ts.predict(X_test_blind)
        
        actual_ts = test_df['Revenue']
        predicted_ts = blind_preds_ts['Revenue']
        
        fold_r2 = r2_score(actual_ts, predicted_ts)
        fold_mae = mean_absolute_error(actual_ts, predicted_ts)
        
        ts_r2_scores.append(fold_r2)
        ts_mae_scores.append(fold_mae)
        
        # Calculate what % of data we trained on for context
        train_pct = (len(train_df) / total_rows) * 100
        print(f"Fold {fold+1} (Trained on {train_pct:.0f}% data): Out-of-Sample R² = {fold_r2:.4f}, MAE = {fold_mae:.4f}")
        
    except Exception as e:
        print(f"Fold {fold+1} encountered a math error: {e}")

# ==========================================================
# FINAL TIME SERIES RESULTS
# ==========================================================
print("\n--- FINAL FORECASTING ACCURACY (ANCHORED TIME SERIES) ---")
if ts_r2_scores:
    print(f"Average Forecasting R²:   {np.mean(ts_r2_scores):.4f} ± {np.std(ts_r2_scores):.4f}")
    print(f"Average Forecasting MAE: {np.mean(ts_mae_scores):.4f} ± {np.std(ts_mae_scores):.4f}")
else:
    print("All folds failed. We need to simplify the model or drop zero-spend variables.")

# %% [markdown]
# ## Model Comparison

# %% [markdown]
# ### Ridge Regression

# %%
print("Running Ridge Regression (ACM Standardized Baseline)...")

# 1. Define Features (X) and Target (y) using ACM Variable Mapping
unique_values = ins_final[ins_final['rval'] != 'Revenue']['rval'].unique()

ridge_features = unique_values

# Note: Using 'Revenue' as the target per your mapping dict
X = us_data[ridge_features]
y = us_data['Revenue']



# 3. Fit Ridge with Cross-Validation
# Automatically selects the best lambda (alpha) to balance bias and variance.
alphas = np.logspace(-3, 3, 100)
ridge_model = RidgeCV(alphas=alphas, cv=5)
ridge_model.fit(X, y)

# 4. Extract and Display Coefficients
ridge_coefs = pd.DataFrame({
    'Feature': ridge_features,
    'Ridge_Coefficient': ridge_model.coef_
}).sort_values(by='Ridge_Coefficient', key=abs, ascending=False)

print(f"Optimal Alpha (Penalty): {ridge_model.alpha_:.3f}")
print(f"R-Squared (Scaled Fit): {ridge_model.score(X, y):.3f}\n")
print("--- RIDGE REGRESSION COEFFICIENTS (ACM STANDARDS) ---")
print(ridge_coefs.to_string(index=False))

# %%
ridge_features

# %%
# # ==============================================================================
# # COMPETING MODEL: THE "FLAT" MLR PROXY (Model M0)
# # Hypothesis: "All channels act independently and directly on Revenue."
# # ==============================================================================

# 1. The Multi Linear Equation
# Every single variable points DIRECTLY to Revenue. No indirect paths.
flat_model_desc = """
    # --- The "Flat" Revenue Equation ---
    
  Revenue ~ Rolling_Revenue_Volatility+ Market_Uncertainty_Index+\
       Market_Disruption_Indicator+ Spend_Meta+ Spend_Shopping+\
       Spend_YouTube+ Spend_Display+ Spend_Search_Brand+\
       Spend_Search_Generic+ Spend_Reddit+ Spend_LinkedIn+\
       Competitor_Spend_Index+ Holiday_Indicator+\
       Budget_Allocation_Instability+ Revenue_Seasonality+\
       Macro_CPI+ Spend_TikTok+ Spend_Affiliate+ Spend_Pinterest+\
       Spend_CTV+ Spend_OLV+ Spend_Audio
       
    # --- Variances (Required for SEM) ---
    Revenue ~~ Revenue
"""

print("Running Competing Model M0 (Flat MLR Proxy)...")
m_flat = semopy.Model(flat_model_desc)
res_flat = m_flat.fit(us_data)
stats_flat = semopy.calc_stats(m_flat)
ins_flat = m_flat.inspect()

print("\n--- FLAT MODEL FIT INDICES ---")
print(stats_flat[['AIC', 'BIC', 'RMSEA', 'CFI']].T)

print("\n--- VERSION 8 (YOUR MODEL) FIT INDICES ---")
# Assuming you still have 'm_v8' or 'm_final' in memory
stats_v8 = semopy.calc_stats(m_final) 
print(stats_v8[['AIC', 'BIC', 'RMSEA', 'CFI']].T)

display(ins_flat)

# %%
# Ensure Graphviz is in PATH (Keep this line as you had it)
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# ==============================================================================
# 4. VISUALIZE THE PATH DIAGRAM
# ==============================================================================
print("Generating Path Diagram...")

# 1. Generate the initial graph object
# We strip the extension here because we will re-render it manually below
g = semopy.semplot(m_flat, "US_SEM_Path_Diagram_Flat.png", plot_covs=True, std_ests=True)

# -------------------------------------------------------
# START: CUSTOMIZE LAYOUT (BALANCE)
# -------------------------------------------------------

# OPTION A: Switch to Left-to-Right orientation (usually better for SEMs)
# Default is 'TB' (Top-to-Bottom). 'LR' often balances wide models better.
g.attr(rankdir='LR')

# OPTION B: Force a square aspect ratio
# 'auto' is default. '1' tries to make it a square.
# 'compress' tries to fit it tight.
# You can try changing this to '0.5' (tall) or '2' (wide) if needed.
g.attr(ratio='1')

# OPTION C: Adjust spacing to make it tighter or looser
# nodesep: space between nodes in the same rank
# ranksep: space between the rows/columns
# g.attr(nodesep='0', ranksep='0.5')

# 2. Re-render the plot with the new attributes
# Note: Graphviz adds the file extension automatically based on 'format'
g.render('US_SEM_Path_Diagram_Flat', format='png', cleanup=True)

# -------------------------------------------------------
# END: CUSTOMIZE LAYOUT
# -------------------------------------------------------

# Display the plot inside the notebook
from IPython.display import Image, display
display(Image("US_SEM_Path_Diagram_Flat.png"))

print("Diagram saved as 'US_SEM_Path_Diagram_Flat.png'")
