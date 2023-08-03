# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:08:30 2023

@author: issaktop
"""

import LilyTools as lt
import pandas as pd
# import matplotlib.pyplot as plt
from datetime import date
# import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pyodbc
import matplotlib.pyplot as plt
from datetime import date, timedelta
#%% EMPS query

def fetch_emps(
        fc_date = date.today() - timedelta(days=1),
        del_start = date(date.today().year, date.today().month, 1),
        del_end = date(date.today().year + 14, 12, 31)
        ):
    """
    Description:
    Query EMPS spot price forecasts for Outright products from Lily
    Monthly delivery granularity
    
    Inputs:
    - fc_date | forecast date | date
    - del_start | delivery start | date
    - del_end | delivery last | date
    Outputs:
    Pandas dataframe containing EMPS spot price forecasts
    - Area | product group | string
    - ValueDate | delivery start | int
    - Variable | scenario label | string
    - Value | forecasted spot price | float
    
    """
    print("Querying LT RiRe EMPS scenarios")
    
    dict_zones = {
        'FI':'MT risk FI power',
        'SE2':'MT risk SE2 power',
        'SE3':'MT risk SE3 power',
        'SYS':'MT risk SYS power'
        }
    
    emps = pd.DataFrame()
    for key_zone, zone in dict_zones.items():
        print(key_zone + '-' + zone)
        df_temp = lt.load_group(
            zone,
            # forecastdate = fc_date,
            timebase='Year',
            minvaluedate=pd.to_datetime(del_start),
            maxvaluedate=pd.to_datetime(del_end)
            )[0]
        df_temp.columns = pd.Series(df_temp.columns).replace({'SYS ':'','FI ':'','SE2 ':'','SE3 ':''}, regex=True)
        df_temp = df_temp.reset_index().melt(id_vars='ValueDate')
        df_temp['Area'] = key_zone
        df_temp = df_temp.reset_index(drop=True)
        emps = pd.concat([emps,df_temp],axis=0)
    emps.columns = ['ValueDate','Variable','Value','Area']
    
    # unify scenario labels
    emps.variable = emps.Variable.str[:4]
    # Aggregation columns
    emps['Delivery year'] = emps.ValueDate.dt.year
    #emps['delqtr'] = "Q"+emps.ValueDate.dt.quarter.astype(str)+"-"+emps.ValueDate.dt.year.astype(str)
    #emps['delmth'] = "M"+emps.ValueDate.dt.month.astype(str)+"-"+emps.ValueDate.dt.year.astype(str)
    return emps
#%%SMFC query

def fetch_smfc(
        fc_date = date.today(),
        del_start = date(date.today().year, date.today().month, 1),
        del_end = date(date.today().year + 14, 12, 31)
        ):
    print("Querying SMFCs")
    """
    Description:
    Query EMPS spot price forecasts for Outright products from Lily
    Monthly delivery granularity
    
    Inputs:
    - fc_date | forecast date | date
    - del_start | delivery start | date
    - del_end | delivery last | date
    Outputs:
    Pandas dataframe containing EMPS spot price forecasts
    - Area | product group | string
    - ValueDate | delivery start | int
    - Value | smoothed valuation price | float
    
    """   
    # SMFC curves (areas)
    dict_smfc_zones = {
        850006003:'FI',
        850006008:'SE2',
        850006007:'SE3',
        850005001:'SYS'
        }
    
    smfc = pd.DataFrame()
    for i in dict_smfc_zones:
        print(dict_smfc_zones[i])
        dfTemp = lt.load_actual(
            i,
            timebase='Year',
            timezone='CET/CEST',
            maxfcst_date=pd.to_datetime(fc_date).strftime('%Y-%m-%d'),
            minvaluedate = pd.to_datetime(del_start),
            maxvaluedate = pd.to_datetime(del_end)
            )
        dfTemp['Area'] = dict_smfc_zones[i]
        smfc = pd.concat([smfc, dfTemp], axis=0)
    smfc.reset_index(inplace=True)
    smfc = smfc.rename(columns={'Value Date':'ValueDate'})
    smfc = smfc.round(2)
    smfc.Value = smfc.Value.astype('float32')
    # Aggregation columns
    smfc['Delivery year'] = smfc.ValueDate.dt.year
    # smfc['delqtr'] = "Q"+smfc.ValueDate.dt.quarter.astype(str)+"-"+smfc.ValueDate.dt.year.astype(str)
    # smfc['delmth'] = "M"+smfc.ValueDate.dt.month.astype(str)+"-"+smfc.ValueDate.dt.year.astype(str)
    return smfc

#%%
emps_original = fetch_emps()
emps = emps_original.copy() #Easier to test different things when reloading EMPS prices isn't necessary every time

# Test plot
import seaborn as sns
sns.set_style('whitegrid')
sns.lineplot(data=emps, x='ValueDate', y='Value', hue='Area', ci=None)

# Weights and transposing
emps['Weight'] = 0
emps.loc[emps['Area'] == 'FI', 'Weight'] = 0.4
emps.loc[emps['Area'] == 'SE3', 'Weight'] = 0.4
emps.loc[emps['Area'] == 'SE2', 'Weight'] = 0.2
emps = emps[emps['Area'] != 'SYS']
emps['Weighted Price'] = emps['Value'] * emps['Weight']
emps = emps.groupby(['Variable', 'Delivery year'])['Weighted Price'].sum().unstack()
emps = emps.T

# Averages and sorting
emps.loc['Column avg'] = emps.mean()
emps = emps.sort_values(by='Column avg', axis=1)
print(emps)


#%%
smfc = fetch_smfc()

# Transforming
smfc = smfc.pivot_table(index='Delivery year', columns=['Area'], values='Value')
smfc = smfc.reset_index()
smfc.drop('SYS', axis=1, inplace=True)

# Weighted price
weights = [0.4, 0.2, 0.4]
smfc[['FI', 'SE2', 'SE3']] = smfc[['FI', 'SE2', 'SE3']].apply(pd.to_numeric)
smfc['Weighted Price'] = (smfc.iloc[:, 1:] * weights).sum(axis=1)

#%%
# Import margins from the excel
margins = pd.read_excel('LT Hedging DEV.xlsx', sheet_name='Margins')
margins = margins.T
margins = margins.set_axis(margins.iloc[0], axis=1)
margins = margins.drop(margins.index[0])

# Inputs
inflation = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
interestrate = [0.035, 0.037, 0.036, 0.035, 0.031, 0.03, 0.029, 0.029, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028, 0.028]
div_payout = 0.75
lt_hr = 0.5

variables = {
    'Year': range(2022, 2038),
    'Inflation': inflation,
    'InterestRate': interestrate,
    'DivPayout': div_payout,
    'LtHr': lt_hr}

inputs = pd.DataFrame(variables)
inputs.set_index('Year', inplace=True)


# Main df
years = list(range(2022, 2038))
financial_metrics = ['Achieved electricity price (€/MWh)', 'Electricity generation (TWh)', 'Electricity revenue',
                     'Other revenue', 'Generation', 'Consumer Solutions', 'City Solutions', 'Others and eliminations',
                     'Total revenue', 'EBITDA', 'Generation E', 'Consumer Solutions E', 'City Solutions E', 'Others and eliminations E',
                     'Tangible assets', 'Depreciation and amortization', 'Financial net debt', 'Interest bearing debt',
                     'Interest rate', 'Net financial items', 'EBIT', 'Taxes', 'Net income', 'FFO', 'Capex', 'Maintenance',
                     'Growth', 'FCF', 'Dividend per share (€)', 'Shares', 'Dividends', 'Debt amortization', 'ND/EBITDA', 'FFO/ND']
df = pd.DataFrame(index=years, columns=financial_metrics)

#%%
print(df.dtypes)
df = df.astype(float)

# Filling the 2022 info as it is
df.loc[2022, 'Tangible assets'] = 10179
df.loc[2022, 'Depreciation and amortization'] = -566
df.loc[2022, 'Financial net debt'] = -794
df.loc[2022, 'Interest bearing debt'] = 6123
df.loc[2022, 'Dividend per share (€)'] = 0.91
df.loc[2022, 'Consumer Solutions'] = 4578
df.loc[2022, 'City Solutions'] = 1282
df['Others and eliminations'] = -1741
df.loc[2022, 'Net financial items'] = -214
df.loc[2022, 'Taxes'] = -249
df.loc[2022, 'EBIT'] = 1611

# More basic inputs
df['Shares'] = 897
df['Growth'] = 0
df.loc[2023, 'Growth'] = -500 #Choose growth investment amount and year
df.loc[2024, 'Growth'] = -500
df.loc[2025, 'Growth'] = -500
df.loc[2028, 'Growth'] = -2000
df['Interest rate'] = inputs.loc[:, 'InterestRate'] # Check these
df['Inflation'] = inputs.loc[:, 'Inflation']

# Choose scenario. Figure out something more sophisticated later on.
scen = emps.iloc[:, 44] # 0 refers to scenario where prices on average are the lowest
df['Achieved electricity price (€/MWh)'] = scen 
prodplans = [42.9, 44.3, 45.5, 45.2, 43.9, 45.2] #Apollo plans from B&W
df.loc[df.index[1:7], 'Electricity generation (TWh)'] = prodplans
realprices = [59.9, 50.8, 46.35, 42.84] #Actual prices from B&W
df.loc[df.index[0:4], 'Achieved electricity price (€/MWh)'] = realprices

# CAPEX: Calculate the new value based on the previous year's value and inflation
df.loc[2022, 'Maintenance'] = -425

for i in range(1, len(df)):
    df.iloc[i, df.columns.get_loc('Maintenance')] = df.iloc[i-1]['Maintenance'] * (1 + df.iloc[i-1]['Inflation']) + (df.iloc[i-1]['Growth'] * 0.032)
    
df['Capex'] = df['Maintenance'] + df['Growth']

# Generation: First 5 years based on production plans
df.loc[2023:2027, 'Electricity generation (TWh)'] = [44.3, 45.5, 45.2, 43.9, 45.2]
df['Electricity revenue'] = df['Achieved electricity price (€/MWh)'] * df['Electricity generation (TWh)']

# Generation from 2028 onwards and then revenue
for year in range(2028, 2038):
    prev_year = year - 1
    df.loc[year, 'Electricity generation (TWh)'] = df.loc[prev_year, 'Electricity generation (TWh)'] - df.loc[prev_year, 'Growth'] / 385

df['Electricity revenue'] = df['Electricity generation (TWh)'] * df['Achieved electricity price (€/MWh)']

# Divisions
df['Generation'] = 1085

for year in range(2023, 2038):
    df.loc[year, 'City Solutions'] = df.loc[year - 1, 'City Solutions'] * (1 + df.loc[year - 1, 'Inflation'])
    df.loc[year, 'Consumer Solutions'] = df.loc[year - 1, 'Consumer Solutions'] * (1 + df.loc[year - 1, 'Inflation'])

df['Other revenue'] = df.iloc[:, 5:8].sum(axis=1)
df['Total revenue'] = df['Electricity revenue'] + df['Other revenue']


# EBITDAs
df.loc[2022, 'Generation E'] = 1765
df.loc[2022, 'Consumer Solutions E'] = 173
df.loc[2022, 'City Solutions E'] = 177
df.loc[2022, 'Others and eliminations E'] = -90

for year in range(2023, 2038):
    df.loc[year, 'Consumer Solutions E'] = df.loc[year, 'Consumer Solutions'] * margins.loc[year, 'Consumer Solutions']
    df.loc[year, 'City Solutions E'] = df.loc[year, 'City Solutions'] * margins.loc[year, 'City Solutions']
    df.loc[year, 'Others and eliminations E'] = df.loc[year, 'Others and eliminations'] * margins.loc[year, 'Others and eliminations']
    df.loc[year, 'Generation E'] = (df.loc[year, 'Generation'] + df.loc[year, 'Electricity revenue']) * margins.loc[year, 'Generation']
    df['EBITDA'] = df.loc[:,'Generation E':'Others and eliminations E'].sum(axis=1)


# Tangible assets in order to calculate other ratios for future years
for year in range(2023, 2038):
    prev_year = year - 1

    # Calculate 'Tangible assets' for the current year
    df.loc[year, 'Tangible assets'] = df.loc[prev_year, 'Tangible assets'] + df.loc[prev_year, 'Depreciation and amortization'] - df.loc[prev_year, 'Capex']

    # Calculate 'Depreciation and amortization' for the current year
    df.loc[year, 'Depreciation and amortization'] = df.loc[year, 'Tangible assets'] * -0.07

df.loc[2022, 'Debt amortization'] = 321
df.loc[2022, 'Net income'] = 996
df.loc[2022, 'FFO'] = 1562
df.loc[2022, 'FCF'] = 1137
df.loc[2022, 'Dividends'] = 816
df.loc[2022, 'ND/EBITDA'] = 0.39
df.loc[2022, 'FFO/ND'] = 1.97


for year in range(2023, 2038):
    prev_year = year - 1
    taxrate = 0.2
    df.loc[year, 'Financial net debt'] = df.loc[prev_year, 'Financial net debt'] - df.loc[prev_year, 'Debt amortization']
    df.loc[year, 'Interest bearing debt'] = df.loc[prev_year, 'Interest bearing debt'] - df.loc[prev_year, 'Debt amortization']
    df.loc[year, 'Net financial items'] = -df.loc[year, 'Interest bearing debt'] * df.loc[year, 'Interest rate']
    df.loc[year, 'EBIT'] = df.loc[year, 'EBITDA'] + df.loc[year, 'Depreciation and amortization'] + df.loc[year, 'Net financial items']
    df.loc[year, 'Taxes'] = -df.loc[year, 'EBIT'] * taxrate
    df.loc[year, 'Net income'] = df.loc[year, 'EBIT'] + df.loc[year, 'Taxes']
    df.loc[year, 'FFO'] = df.loc[year, 'Net income'] - df.loc[year, 'Depreciation and amortization']
    df.loc[year, 'FCF'] = df.loc[year, 'FFO'] + df.loc[year, 'Capex']
    df.loc[year, 'Dividends'] = df.loc[year, 'Dividend per share (€)'] * df.loc[year, 'Shares']
    df.loc[year, 'Debt amortization'] = df.loc[year, 'FCF'] - df.loc[year, 'Dividends']
    df.loc[year, 'Dividends'] = max(df.loc[year, 'Net income'] * div_payout, 0)
    df.loc[year, 'Debt amortization'] = df.loc[year, 'FCF'] - df.loc[year, 'Dividends']
    df.loc[year, 'Dividend per share (€)'] = df.loc[year, 'Dividends'] / df.loc[year, 'Shares']
    df.loc[year, 'ND/EBITDA'] = df.loc[year, 'Financial net debt'] / df.loc[year, 'EBITDA']
    df.loc[year, 'FFO/ND'] = df.loc[year, 'FFO'] / df.loc[year, 'Financial net debt']
    

#%%

# Define the function to calculate the ratios
def financials(scenario_prices, model_df):
    results = {}

    years = list(range(2023, 2038))

    # Iterate over scenarios
    for scenario in scenario_prices.columns:
        temp_df = model_df.copy()
        temp_df['Price'] = scenario_prices[scenario].values

        div_ratios, ndebitda_ratios, ffond_ratios = [], [], []

        for year in years:
            if year in temp_df.index:
                div_ratios.append(temp_df.loc[year, 'Dividend per share (€)'])
                ndebitda_ratios.append(temp_df.loc[year, 'ND/EBITDA'])
                ffond_ratios.append(temp_df.loc[year, 'FFO/ND'])
            else:
                div_ratios.append(None)
                ndebitda_ratios.append(None)
                ffond_ratios.append(None)
        results[scenario] = {
            'Dividend': div_ratios,
            'ND/EBITDA': ndebitda_ratios,
            'FFO/ND': ffond_ratios
        }

    return results


results = financials(emps, df)
results_df = pd.DataFrame.from_dict({(i,j): results[i][j]
                           for i in results.keys()
                           for j in results[i].keys()})
results_df.index = pd.Index(range(2023, 2038), name='Year')

print(results_df)

#%%
# Define the function to calculate the ratios
import pandas as pd

# Define the function to calculate the ratios
def financials(scenario_prices, model_df):
    results = {}

    # Get a list of years from 2023 to 2037
    years = list(range(2023, 2038))

    # Transpose the scenario_prices DataFrame so that scenarios are rows and years are columns
    scenario_prices = scenario_prices.T

    # Iterate over each row (scenario) in the scenario_prices DataFrame
    for scenario in scenario_prices.index:
        # Create a copy of the model_df to avoid modifying the original data
        temp_df = model_df.copy()

        # Apply the scenario prices to the temporary DataFrame
        temp_df['Price'] = scenario_prices.loc[scenario].values

        # Initialize empty lists to store the ratios for each year
        div_ratios, ndebitda_ratios, ffond_ratios = [], [], []

        # Loop through each year and calculate the ratios
        for year in years:
            if year in temp_df.index:
                # Fetch the corresponding values for each year and ratio
                dividend = temp_df.loc[year, 'Dividend per share (€)']
                ndebitda = temp_df.loc[year, 'ND/EBITDA']
                ffond = temp_df.loc[year, 'FFO/ND']

                # Append the values to the respective lists
                div_ratios.append(dividend)
                ndebitda_ratios.append(ndebitda)
                ffond_ratios.append(ffond)
            else:
                # If data is not available for a specific year, append None
                div_ratios.append(None)
                ndebitda_ratios.append(None)
                ffond_ratios.append(None)

        # Append the scenario and the lists of ratios to the results dictionary
        results[scenario] = {
            'Dividend': div_ratios,
            'ND/EBITDA': ndebitda_ratios,
            'FFO/ND': ffond_ratios
        }

    return results

# Example usage
# Assuming emps and df are the DataFrames you mentioned before
results = financials(emps, df)

# Convert the results dictionary to a DataFrame with years as the index and scenarios as columns
results_df = pd.DataFrame.from_dict(results, orient='index')
results_df.index = pd.Index(range(2023, 2038), name='Year')

print(results_df)


sample_emps = emps.sample(n=5).copy()  # Replace 5 with the desired number of rows to copy
sample_df = df.sample(n=5).copy()      # Replace 5 with the desired number of rows to copy

# Print the DataFrames
print("Sample of emps DataFrame:")
print(sample_emps)
print("\nSample of df DataFrame:")
print(sample_df)
