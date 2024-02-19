# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 19:31:01 2023

@author: issaktop
"""

import LilyTools as lt
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import pyodbc
from datetime import date, timedelta
import seaborn as sns

#%% EMPS 
def fetch_emps(
        fc_date = date(2024, 2, 13) + timedelta(days=1),
        del_start = date(date.today().year, date.today().month, 1),
        del_end = date(date.today().year + 20, 12, 31)
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
        'FI':'MT risk 77-21 FI power',
        'SE2':'MT risk 77-21 SE2 power',
        'SE3':'MT risk 77-21 SE3 power',
        'SYS':'MT risk 77-21 SYS power'
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
    
    # Unify scenario labels
    emps.variable = emps.Variable.str[:4]
    # Aggregation columns
    emps['Delivery year'] = emps.ValueDate.dt.year
    return emps

#%% SMFC
def fetch_smfc(
        fc_date = date(2024, 2, 13),
        del_start = date(date.today().year, date.today().month, 1),
        del_end = date(date.today().year + 20, 12, 31)
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
        850006007:'SE3'
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


#%% Fetching EMPS and SMFC prices

a_emps = fetch_emps()
a_smfc = fetch_smfc()

emps = a_emps.copy()
smfc = a_smfc.copy()
hedgeprice = '-15 %'

#%% Modifying EMPS df

def modify_emps(emps: pd.DataFrame(),
                fi_weight: int,
                sto_weight: int,
                sun_weight: int,
                discount_factor: int):
    """
    Calculates the weighted average of EMPS prices. Weights for each price area are given as inputs.
    EMPS prices are also discoutned (discount_factor) to give more weigth to front year prices.

    """
# EMPS: Weights and transposing
    emps['Weight'] = 0
    emps.loc[emps['Area'] == 'FI', 'Weight'] = fi_weight
    emps.loc[emps['Area'] == 'SE3', 'Weight'] = sto_weight
    emps.loc[emps['Area'] == 'SE2', 'Weight'] = sun_weight
    emps = emps[emps['Area'] != 'SYS']
    emps['Weighted Price'] = emps['Value'] * emps['Weight']
    emps = emps.groupby(['Variable', 'Delivery year'])['Weighted Price'].sum().unstack()
    emps = emps.T
    
    # EMPS: Discounting
    years = list(range(2024, 2045))
    disc = pd.DataFrame(index=years)
    disc['Discounter'] = discount_factor
    disc['Discount factor'] = (1 / disc['Discounter'])**(disc.index - 2022)
    emps_temp = emps.multiply(disc['Discount factor'], axis=0)
    
    # EMPS: Averages, sorting, and undiscounting so that the df has right order
    emps_temp.loc['Column avg'] = emps_temp.mean()
    emps_temp = emps_temp.sort_values(by='Column avg', axis=1)
    emps_temp = emps_temp.drop('Column avg', axis=0)
    undiscounted_emps = emps_temp.divide(disc['Discount factor'], axis=0)
    undiscounted_emps.describe()
    emps = undiscounted_emps
    return emps

#%% Modifying SMFC df

def modify_smfc(smfc: pd.DataFrame(),
                weight_list: list):
    """
    Calculates the weighted average of SMFC prices. This is based on a 
    list (order: FI, SE3, SE2)
    """
# SMFC: Transforming
    smfc = smfc.pivot_table(index='Delivery year', columns=['Area'], values='Value')
    smfc = smfc.reset_index()
    # Extending until 2044
    last_year = smfc[smfc['Delivery year'] == 2037].iloc[0]
    missing = pd.DataFrame([last_year] * 7)
    missing['Delivery year'] = range(2038, 2045)
    smfc = pd.concat([smfc, missing], ignore_index=True)
    # SMFC: Weighted price
    weights = weight_list
    smfc[['FI', 'SE3', 'SE2']] = smfc[['FI', 'SE3', 'SE2']].apply(pd.to_numeric)
    smfc['Weighted Price'] = (smfc.iloc[:, 1:] * weights).sum(axis=1)
    smfc = smfc.astype(float)
    return smfc

#%% Importing inputs

def import_inputs(excel_name:str):
    """
    Imports two sheets from input Excel-file. Imported sheets are the input template
    and margins estimates.
    """
    
    margins = pd.read_excel(excel_name, sheet_name='margins', index_col=0)
    margins = margins.T
    margins = margins.reset_index()
    margins = margins.rename(columns = {'index':'Year'})
    
    df = pd.read_excel(excel_name, sheet_name='inputs')
    df = df.astype(float)
    
    return margins, df

# Setting initial LT HR and dividend payout ratio
variables = {
    'Year': range(2023, 2045),
    'DivPayout': 0.75,
    'LtHr': 0.1}

inputs = pd.DataFrame(variables)
inputs.set_index('Year', inplace=True)

margins, df = import_inputs(r'C:\Users\issaktop\OneDrive - Fortum\Documents\Python\ALL\LT Hedging\input_df.xlsx')
emps = modify_emps(emps, 0.4, 0.4, 0.2, 1.06)
smfc = modify_smfc(smfc, [0.4, 0.4, 0.2])

#%% Filling the df 

for year in range(1, len(df)):
    prev_year = year - 1
    taxrate = 0.2
    div_payout = inputs.loc[2023, 'LtHr']
    df.iloc[year, df.columns.get_loc('Maintenance')] = df.iloc[prev_year]['Maintenance'] * (1 + df.iloc[prev_year]['Inflation']) + (df.iloc[prev_year]['Growth'] * 0.032)    
    df['Capex'] = df['Maintenance'] + df['Growth']
    df.loc[year, 'Electricity generation (TWh)'] = df.loc[prev_year, 'Electricity generation (TWh)'] - df.loc[prev_year, 'Growth'] / 385
    df.loc[year, 'Electricity revenue'] = df.loc[year, 'Electricity generation (TWh)'] * df.loc[year, 'Price (€/MWh)']
    df.loc[year, 'CiS'] = df.loc[prev_year, 'CiS'] * (1 + df.loc[prev_year, 'Inflation'])
    df.loc[year, 'CoS'] = df.loc[prev_year, 'CoS'] * (1 + df.loc[prev_year, 'Inflation'])
    df['Other revenue'] = df.iloc[:, 5:8].sum(axis=1)
    df['Total revenue'] = df['Electricity revenue'] + df['Other revenue']
    df.loc[year, 'Tangible assets'] = df.loc[prev_year, 'Tangible assets'] + df.loc[prev_year, 'Depreciation and amortization'] - df.loc[prev_year, 'Capex']
    df.loc[year, 'Depreciation and amortization'] = df.loc[year, 'Tangible assets'] * -0.07
    df.loc[year, 'CoS E'] = df.loc[year, 'CoS'] * margins.loc[year, 'Consumer Solutions']
    df.loc[year, 'CiS E'] = df.loc[year, 'CiS'] * margins.loc[year, 'City Solutions']
    df.loc[year, 'Others E'] = df.loc[year, 'Others'] * margins.loc[year, 'Others and eliminations']
    df.loc[year, 'Generation E'] = (df.loc[year, 'Generation'] + df.loc[year, 'Electricity revenue']) * margins.loc[year, 'Generation']
    df['EBITDA'] = df.loc[:,'Generation E':'Others E'].sum(axis=1)
    df.loc[year, 'Financial net debt'] = df.loc[prev_year, 'Financial net debt'] - df.loc[prev_year, 'Debt amortization']
    df.loc[year, 'Interest bearing debt'] = df.loc[prev_year, 'Interest bearing debt'] - df.loc[prev_year, 'Debt amortization']
    df.loc[year, 'Net financial items'] = -df.loc[year, 'Interest bearing debt'] * df.loc[year, 'Interest rate']
    df.loc[year, 'EBT'] = df.loc[year, 'EBITDA'] + df.loc[year, 'Depreciation and amortization'] + df.loc[year, 'Net financial items']
    df.loc[year, 'Taxes'] = -df.loc[year, 'EBT'] * taxrate
    df.loc[year, 'Net income'] = df.loc[year, 'EBT'] + df.loc[year, 'Taxes']
    df.loc[year, 'FFO'] = df.loc[year, 'Net income'] - df.loc[year, 'Depreciation and amortization']
    df.loc[year, 'FCF'] = df.loc[year, 'FFO'] + df.loc[year, 'Capex']
    df.loc[year, 'Dividends'] = max(df.loc[year, 'Net income'] * div_payout, 0)
    df.loc[year, 'Debt amortization'] = df.loc[year, 'FCF'] - df.loc[year, 'Dividends']
    df.loc[year, 'Dividend per share (€)'] = df.loc[year, 'Dividends'] / df.loc[year, 'Shares']
    df.loc[year, 'ND/EBITDA'] = df.loc[year, 'Financial net debt'] / df.loc[year, 'EBITDA']
    df.loc[year, 'FFO/ND'] = df.loc[year, 'FFO'] / df.loc[year, 'Financial net debt']
    
df.set_index('Year', inplace=True)
df.index = df.index.astype(int)

#%%

# User-specified values for price and hedge ratio for these years, CALCULATED IN SEPERATE EXCEL
user_values = {
    2024: [46.49, 0.75],
    2025: [41.57, 0.75],
    2026: [37.03, 0.75],  
    2027: [37.24, 0.75],  
}

def power_price(lthr:float,
        smfc:pd.DataFrame(),
        emps:pd.DataFrame(),
        user_values:list):
    """
    Calculates the achieved power price based on EMPS and SMFC prices. In addition,
    a dictionary with user-given values works as an input for front years.

    """
    # Unhedged part is EMPS and hedge price is SMFC.
    lt_hr = pd.DataFrame(inputs['LtHr'])
    lt_hr['LtHr'] = lthr
    lt_hr['Unhedged'] = 1 - lt_hr['LtHr']  
    lt_hr = lt_hr.reset_index()
    smfc.rename(columns={'Delivery year': 'Year'}, inplace=True)
    lt_hr = lt_hr.merge(smfc, on='Year')
    lt_hr = lt_hr.drop(columns=['FI', 'SE2', 'SE3'])
    lt_hr.rename(columns={'Weighted Price': 'SMFC'}, inplace=True)
    
    # Using values defined in the user_values dict to get the "correct" prices for front years
    for year, values in user_values.items():
        lt_hr.loc[lt_hr['Year'] == year, ['SMFC', 'LtHr']] = values
        
    for year, values in user_values.items():
        lt_hr.loc[lt_hr['Year'] == year, 'Unhedged'] = 1 - values[1]
    
    lt_hr['Hedge price'] = lt_hr['LtHr'] * lt_hr['SMFC']
    lt_hr.set_index('Year', inplace=True)
    
    emps_multiplied = emps.reset_index()
    emps_multiplied.rename(columns={'Delivery year': 'Year'}, inplace=True)
    emps_multiplied.set_index('Year', inplace=True)
    emps_multiplied = emps_multiplied.multiply(lt_hr['Unhedged'], axis='index')
    
    lt_hr = lt_hr.merge(emps_multiplied, on='Year')
    cols = lt_hr.columns[4:]
    lt_hr[cols] = lt_hr[cols].add(lt_hr['Hedge price'], axis='index')
    lt_hr = lt_hr.iloc[:, 4:]

    return lt_hr

app = power_price(0.2, smfc, emps, user_values)

#%%

def stats_to_emps(emps_df:pd.DataFrame,
                  percentiles:list):
    """
    Calculates the mean and given percentiles of EMPS prices.

    """
    scen_mean = emps_df.mean(axis=1)
    
    scen_perc = emps_df.quantile(q = percentiles, axis=1)
    scen_perc = scen_perc.T

    stats = pd.concat([scen_mean, scen_perc], axis=1)
    stats.columns = ['Mean'] + [f'{int(p*100)}th percentile' for p in percentiles]
    stats = pd.DataFrame(stats)

    emps_df = pd.merge(emps_df, stats, left_index=True, right_index=True)    
    return emps_df

emps = stats_to_emps(emps, percentiles = [0.2, 0.8])

#%%

margins.set_index('Year', inplace=True)

def calculate_ratios(scenarios:pd.DataFrame, model:pd.DataFrame,
                     prodlist:list, div_payout:float,
                     hedge_price = str):
    """
    Loops through the calculation template and saves the financial ratios per 
    scenario into a dictionary.

    """
    
    results = {'Scenario': [], 'Year': [], 'Dividend per share (€)': [], 'ND/EBITDA': [], 'FFO/ND': []}

    # Iterate over scenarios
    for scenario in scenarios.columns:  
        scenario_prices= scenarios[scenario]
        
        # 2022 figures ready
        for year in model.index:
            if year < 2023:
                continue

            if year in model.index and year in scenario_prices.index:
                prev_year = year - 1
                taxrate = 0.2
                
                model['Price (€/MWh)'] = scenario_prices
                model.loc[2024:2028, 'Electricity generation (TWh)'] = prodlist 
                model['Electricity revenue'] = model['Price (€/MWh)'] * model['Electricity generation (TWh)']
                model.loc[year, 'Electricity generation (TWh)'] = model.loc[prev_year, 'Electricity generation (TWh)'] - model.loc[prev_year, 'Growth'] / 385
                model['Electricity revenue'] = model['Electricity generation (TWh)'] * model['Price (€/MWh)']
                model.loc[year, 'CiS'] = model.loc[prev_year, 'CiS'] * (1 + model.loc[prev_year, 'Inflation'])
                model.loc[year, 'CoS'] = model.loc[prev_year, 'CoS'] * (1 + model.loc[prev_year, 'Inflation'])
                model['Other revenue'] = model.iloc[:, 5:8].sum(axis=1)
                model['Total revenue'] = model['Electricity revenue'] + model['Other revenue']
                model.loc[year, 'CoS E'] = model.loc[year, 'CoS'] * margins.loc[year, 'Consumer Solutions']
                model.loc[year, 'CiS E'] = model.loc[year, 'CiS'] * margins.loc[year, 'City Solutions']
                model.loc[year, 'Others E'] = model.loc[year, 'Others'] * margins.loc[year, 'Others and eliminations']
                model.loc[year, 'Generation E'] = (model.loc[year, 'Generation'] + model.loc[year, 'Electricity revenue']) * (0.0122 * (model.loc[year, 'Price (€/MWh)'] - 5.7))
                model['EBITDA'] = model.loc[:, 'Generation E':'Others E'].sum(axis=1)
                model.loc[year, 'Financial net debt'] = model.loc[prev_year, 'Financial net debt'] - model.loc[prev_year, 'Debt amortization']
                model.loc[year, 'Interest bearing debt'] = model.loc[prev_year, 'Interest bearing debt'] - model.loc[prev_year, 'Debt amortization']
                model.loc[year, 'Net financial items'] = -model.loc[year, 'Interest bearing debt'] * model.loc[year, 'Interest rate']
                model.loc[year, 'EBT'] = model.loc[year, 'EBITDA'] + model.loc[year, 'Depreciation and amortization'] + model.loc[year, 'Net financial items']                
                
                if model.loc[year, 'EBT'] > 0:
                    model.loc[year, 'Taxes'] = -model.loc[year, 'EBT'] * taxrate
                else:
                    model.loc[year, 'Taxes'] = 0
    
                model.loc[year, 'Net income'] = model.loc[year, 'EBT'] + model.loc[year, 'Taxes']
                model.loc[year, 'FFO'] = model.loc[year, 'Net income'] - model.loc[year, 'Depreciation and amortization']
                model.loc[year, 'FCF'] = model.loc[year, 'FFO'] + model.loc[year, 'Capex']
                model.loc[year, 'Dividends'] = max(model.loc[year, 'Net income'] * div_payout, 0)
                model.loc[year, 'Debt amortization'] = model.loc[year, 'FCF'] - model.loc[year, 'Dividends']
                model.loc[year, 'Dividend per share (€)'] = model.loc[year, 'Dividends'] / model.loc[year, 'Shares']
                model.loc[year, 'ND/EBITDA'] = model.loc[year, 'Financial net debt'] / model.loc[year, 'EBITDA']
                model.loc[year, 'FFO/ND'] = model.loc[year, 'FFO'] / model.loc[year, 'Financial net debt']
    
                div = model.loc[year, 'Dividend per share (€)']
                ndebitda = model.loc[year, 'ND/EBITDA']
                ffond = model.loc[year, 'FFO/ND']
                results['Scenario'].append(scenario)
                results['Year'].append(year)
                results['Dividend per share (€)'].append(div)
                results['ND/EBITDA'].append(ndebitda)
                results['FFO/ND'].append(ffond)
                results['Hedge price'] = hedge_price

    results_df = pd.DataFrame(results)
    return results_df

prodlist = [46.312, 46.37, 45.301, 46.366, 46.358]
results_df = calculate_ratios(app, df, prodlist, 0.75, hedgeprice)
print(results_df.describe())

#%% Running the model with different long-term hedge ratios

percentages = [0.1, 0.2, 0.3]
results_dict = {}

# Loop through the percentage values
for percentage in percentages:
    app = power_price(percentage, smfc, emps, user_values)
    results_dict[f'App_{int(percentage*100)}'] = app
    
    res = calculate_ratios(app, df, prodlist, 0.75, hedgeprice)
    results_dict[f'Results_{int(percentage*100)}'] = res

#%%

def extract_df(result_dict:dict,
               df_name:pd.DataFrame):
    df = results_dict[df_name]
    return df

def df_filtering(data_frame, scenarios):
    return data_frame[data_frame['Scenario'].isin(scenarios)]

results_df_filt = df_filtering(extract_df(results_dict, 'Results_10'),
                               scenarios = ['20th percentile', 'Mean', '80th percentile'])

def generate_plots(data_frame, cols, scenarios):
    for column in cols:
        sns.set(style='whitegrid', context='talk')
        plt.figure(figsize=(12, 7), dpi=100)
        sns.lineplot(data=data_frame, x='Year', y=column, hue='Scenario', marker='o',
                     linewidth=2, palette=sns.color_palette("hls", len(scenarios)))
        plt.legend()
        plt.title(f'{column} Over Time for Each Scenario')
        plt.xlabel('Year')
        plt.ylabel(column)
        plt.grid(True)
        plt.xticks(data_frame['Year'].unique().astype(int))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

generate_plots(results_df_filt,
               cols = ['ND/EBITDA', 'Dividend per share (€)'],
               scenarios = ['20th percentile', 'Mean', '80th percentile'])

#%% Excels

df = df.dropna()

with pd.ExcelWriter('lt_hedging_outputs.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='Results', index=True)
    emps.to_excel(writer, sheet_name='EMPS', index=True)
    smfc.to_excel(writer, sheet_name='SMFC', index=True)
    df.to_excel(writer, sheet_name='DF', index=True)
    
with pd.ExcelWriter('financial_ratios_output.xlsx') as writer:
    for sheet_name, results_df in results_dict.items():
        results_df.to_excel(writer, sheet_name=sheet_name, index=True)
        
#%%

plt.figure(figsize=(9,7), dpi=100)
sns.boxplot(data=results_df, x="Year", y="ND/EBITDA")
plt.title("Boxplots of ND/EBITDA by Year")
plt.xlabel("Year")
plt.ylabel("ND/EBITDA")
plt.xticks(fontsize=11)
plt.xticks(rotation=45) 
plt.show()

plt.figure(figsize=(9,7), dpi=100)
sns.boxplot(data=results_df, x="Year", y="Dividend per share (€)")
plt.title("Boxplots of Dividend per Share by Year")
plt.xlabel("Year")
plt.ylabel("Dividend")
plt.xticks(fontsize=11) 
plt.xticks(rotation=45)
plt.show()
