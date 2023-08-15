# -*- coding: utf-8 -*- 
# @Time : 2023/5/1
# @Author : Slim
# @File : TOPSIS_alpha_newbility.py

import pandas as pd
from TOPSIS import topsis
import matplotlib.pyplot as plt
import numpy as np

# Parameters
day = 3
weight = 1/day
number_of_stocks_per_group = 10
top_percentage = 0.5
start_date = '20180101'
end_date = '20230101'

# Settings
initial_asset = 1e6
bottom_percentage = 1-top_percentage

profitability = {
    'base':[],
    'top':[],
    'bottom':[],
    'all':[]
}

profit = {
    'base':[],
    'top':[],
    'bottom':[],
    'all':[initial_asset]
}

percentage_dict = {}
# 1. Data preprocessing/preparation
# 1.1 Get historical price data
df_price = pd.read_csv('166price.csv',index_col=0)
df_price = df_price.sort_index(ascending=True)

# 1.2 Compute rate of return compared to day before
df_return = (df_price/df_price.shift(1)-1)*100
df_return = df_return.dropna()
df_return = df_return.round(8)
df_return = df_return.replace(0,0.00000001)

df_price = df_price.T
df_return = df_return.T

# 1.3 Compute the rank according to TOPSIS based on the previous n days
df_rank = pd.DataFrame()
for i in range(day-1,len(df_return.columns)):
    data = df_return.iloc[:,i-(day-1):i+1]
    df_rank[df_return.columns[i]] = topsis(data,weight)
print('Data preparation complete!')


# 2. Define functions for trading
# 2.1. Get the top group and bottom group, consisting of stock codes, for a given day
def get_top_bottom(num_of_stocks, day):
    top_group = df_rank[day][df_rank[day]<=num_of_stocks].index.tolist()
    bottom_group = df_rank[day][df_rank[day] > len(df_rank)-num_of_stocks].index.tolist()
    return top_group,bottom_group

# 2.2. Get the price of the given group for a given day
def get_price_list(group,day):
    price_list = []
    for stock in group:
        price_list.append(df_price.loc[stock,day])
    return price_list

# 2.3. "Evolutionary" part: compute the percentage change of portfolio for next time period
def compute_top_bottom_percentage_change(top_return, bottom_return):
    return (top_return-bottom_return)/500

# 2.4. Takes in a list consisting of prices (can get from get_price_list), and do buy stock operation
def buy_stock(val_sr, index_group, asset):
    cost = sum(val_sr[index] for index in index_group)
    hold = asset // cost
    remain_asset = asset - (hold * cost)
    return hold, remain_asset

# 2.5. Takes in a list consisting of prices (can get from get_price_list), and do sell stock operation
def sell_stock(val_sr, index_group, hold):
    price = sum(val_sr[index] for index in index_group)
    new_asset_from_sell = price * hold
    return new_asset_from_sell

# 2.6. Compute the profitability/rate of return (%) compared to the previous time period (not the initial state)
def compute_profitability(initial_asset, new_asset):
    return 100*(new_asset - initial_asset) / initial_asset


# 3. Backtesting for profitability and percentage_change
for day in df_rank:
    if day == df_rank.columns[0]:  # If the first trade date, only allow buy
        # top
        top_group, bottom_group = get_top_bottom(number_of_stocks_per_group, day)
        hold_top, remain_asset_top = buy_stock(df_price[day], top_group, initial_asset*top_percentage)
        # bottom
        hold_bottom, remain_asset_bottom = buy_stock(df_price[day], bottom_group, initial_asset*bottom_percentage)
        hold_base, remain_asset_base = buy_stock(df_price[day],df_rank.index.tolist(),initial_asset)

        # Since it's the first day, profitability all sets to 0
        profitability['base'].append(0)
        profitability['top'].append(0)
        profitability['bottom'].append(0)
        profitability['all'].append(0)

        # set the two variables we will use later to the same value of initial_asset
        new_asset_all = initial_asset
        previous_day_base = initial_asset

    # If not the first day, then firstly sell all the stocks from yesterday and compute profitability
    # This would involve 4 groups - base, top, bottom, and all (aggregate top and bottom)
    # Then we will get the top group and bottom group of the new day (today)
    # And then buy stocks according to the groups
    else:
        # Sell, according to the group by the t-1 period's result
        # base case
        new_asset_from_sell_base = sell_stock(df_price[day], df_rank.index.tolist(), hold_base)
        new_asset_base = new_asset_from_sell_base + remain_asset_base
        profitability['base'].append(compute_profitability(previous_day_base,new_asset_base))
        previous_day_base = new_asset_base

        # top
        new_asset_from_sell_top = sell_stock(df_price[day], top_group, hold_top)
        new_asset_top = new_asset_from_sell_top + remain_asset_top
        top_return = sum(df_return.loc[stock,day] for stock in top_group)
        profitability['top'].append(compute_profitability(new_asset_all*top_percentage, new_asset_top))

        #  bottom
        new_asset_from_sell_bottom = sell_stock(df_price[day], bottom_group, hold_bottom)
        new_asset_bottom = new_asset_from_sell_bottom + remain_asset_bottom
        bottom_return = sum(df_return.loc[stock,day] for stock in bottom_group)
        profitability['bottom'].append(compute_profitability(new_asset_all*bottom_percentage, new_asset_bottom))

        # overall
        profitability['all'].append(compute_profitability(new_asset_all,new_asset_top+new_asset_bottom))
        new_asset_all = new_asset_top+new_asset_bottom
        # compute profit of all here, because in the next step for profit calculation, we'll not use the portfolio as here
        profit['all'].append(new_asset_all)


        # Change the percentage
        percentage_dict[day] = [top_percentage,bottom_percentage]
        percentage_change = compute_top_bottom_percentage_change(top_return,bottom_return)
        top_percentage += percentage_change
        # Set boundaries, so that the two strategies won't "kill" each other
        if top_percentage>=0.999:
            top_percentage = 0.999
        if top_percentage <=0.001:
            top_percentage = 0.001
        bottom_percentage = 1-top_percentage

        # Buy again
        # change the target group first
        top_group, bottom_group = get_top_bottom(number_of_stocks_per_group, day)
        # and then buy
        new_asset_top = new_asset_all*top_percentage
        new_asset_bottom = new_asset_all*bottom_percentage
        hold_top, remain_asset_top = buy_stock(df_price[day], top_group, new_asset_all*top_percentage)
        hold_bottom, remain_asset_bottom = buy_stock(df_price[day], bottom_group, new_asset_all*bottom_percentage)
        hold_base, remain_asset_base = buy_stock(df_price[day], df_rank.index.tolist(), new_asset_base)

profitability = pd.DataFrame(profitability,index=df_rank.columns)
percentage_df = pd.DataFrame(percentage_dict,index = ['top','bottom']).T

print('profitability:')
print(profitability)
print()
print('percentage_df')
print(percentage_df)
print()

# 4. Backtesting for profit (the total asset, not just the profit, we obtain during each time period)
for day in df_rank:
    # Pretty much the same code as before, except that this time we don't need to do portfolio
    # So all percentages are not used here
    if day == df_rank.columns[0]:
        # top
        top_group, bottom_group = get_top_bottom(number_of_stocks_per_group, day)
        hold_top, remain_asset_top = buy_stock(df_price[day], top_group, initial_asset)
        # bottom
        hold_bottom, remain_asset_bottom = buy_stock(df_price[day], bottom_group, initial_asset)
        hold_base, remain_asset_base = buy_stock(df_price[day], df_rank.index.tolist(), initial_asset)

        profit['base'].append(initial_asset)
        profit['top'].append(initial_asset)
        profit['bottom'].append(initial_asset)
        new_asset_all = initial_asset

    else:
        # base case
        new_asset_from_sell_base = sell_stock(df_price[day], df_rank.index.tolist(), hold_base)
        new_asset_base = new_asset_from_sell_base + remain_asset_base
        profit['base'].append(new_asset_base)

        # top
        new_asset_from_sell_top = sell_stock(df_price[day], top_group, hold_top)
        new_asset_top = new_asset_from_sell_top + remain_asset_top
        profit['top'].append(new_asset_top)

        #  bottom
        new_asset_from_sell_bottom = sell_stock(df_price[day], bottom_group, hold_bottom)
        new_asset_bottom = new_asset_from_sell_bottom + remain_asset_bottom
        profit['bottom'].append(new_asset_bottom)

        # Buy again
        top_group, bottom_group = get_top_bottom(number_of_stocks_per_group, day)
        hold_top, remain_asset_top = buy_stock(df_price[day], top_group, new_asset_top)
        hold_bottom, remain_asset_bottom = buy_stock(df_price[day], bottom_group, new_asset_bottom)
        hold_base, remain_asset_base = buy_stock(df_price[day], df_rank.index.tolist(), new_asset_base)

profit = pd.DataFrame(profit,index=df_rank.columns)
profit = profit/1e6
print('profit(1 million yuan):')
print(profit)

# 5. Visualization
profitability.index = pd.to_datetime(profitability.index.to_numpy(str))
profitability.loc[start_date:end_date].plot(color=["blue","orange","green","red"])
plt.xticks(rotation=45)
plt.ylabel('Profitability (%)')
plt.xlabel('Trade date (Year)')
plt.title('Profitability Overtime')
plt.show()

profitability.index = pd.to_datetime(profitability.index.to_numpy(str))
profitability["base"].loc[start_date:end_date].plot(color="blue")
plt.xticks(rotation=45)
plt.ylabel('Profitability (%)')
plt.xlabel('Trade date (Year)')
plt.title('Base Profitability Overtime')
plt.show()


profitability.index = pd.to_datetime(profitability.index.to_numpy(str))
profitability["top"].loc[start_date:end_date].plot(color="orange")
plt.xticks(rotation=45)
plt.ylabel('Profitability (%)')
plt.xlabel('Trade date (Year)')
plt.title('Top {} Profitability Overtime'.format(number_of_stocks_per_group))
plt.show()


profitability.index = pd.to_datetime(profitability.index.to_numpy(str))
profitability["bottom"].loc[start_date:end_date].plot(color="green")
plt.xticks(rotation=45)
plt.ylabel('Profitability (%)')
plt.xlabel('Trade date (Year)')
plt.title('Bottom {} Profitability Overtime'.format(number_of_stocks_per_group))
plt.show()


profitability.index = pd.to_datetime(profitability.index.to_numpy(str))
profitability["all"].loc[start_date:end_date].plot(color="red")
plt.xticks(rotation=45)
plt.ylabel('Profitability (%)')
plt.xlabel('Trade date (Year)')
plt.title('All Profitability Overtime')
plt.show()

profit.index = pd.to_datetime(profit.index.to_numpy(str))
profit.loc[start_date:end_date].plot()
plt.xticks(rotation=45)
plt.ylabel('Profit (1 million yuan)')
plt.xlabel('Trade date (Year)')
plt.title('Profit Overtime')
plt.show()

percentage_df.index = pd.to_datetime(percentage_df.index.to_numpy(str))
percentage_df.loc[start_date:end_date].plot()
plt.xticks(rotation=45)
plt.ylabel('Share Allocated')
plt.xlabel('Trade date (Year)')
plt.title('Evolutionary Share Overtime')
plt.show()


# 6. Report statistics 
# define parameters 
N = 252 #255 trading days in a year
rf =0.01 #1% risk free rate

# define formulas/functions for ratios 

def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N -rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(series, N,rf):
    mean = series.mean() * N -rf
    std_neg = series[series<0].std()*np.sqrt(N)
    return mean/std_neg

def max_drawdown(return_series):
    comp_ret = (return_series+1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret/peak)-1
    return dd.min()

# Sharpe Ratio
print("Sharpe Ratio")
print(profitability.mean(axis=0)/profitability.std(axis=0))
sharpes = profitability.apply(sharpe_ratio, args=(N,rf,),axis=0)

print("Annualized Sharpe Ratio")
print(sharpes)
plt.ylabel('Sharpe ratio')
sharpes.plot.bar()
plt.show()

# Sortino Ratio
sortinos = profitability.apply(sortino_ratio, args=(N,rf,), axis=0 )
sortinos.plot.bar()
plt.ylabel('Sortino Ratio')
plt.show()

# Max Drawdown
max_drawdowns = profitability.apply(max_drawdown,axis=0)
max_drawdowns.plot.bar()
plt.ylabel('Max Drawdown')
plt.show()

# Calmar Ratio
calmars = profitability.mean()*N/abs(max_drawdowns)
calmars.plot.bar()
plt.ylabel('Calmar ratio')
plt.show()

# Ratio Summary tables
btstats = pd.DataFrame()
btstats['sortino'] = sortinos
btstats['sharpe'] = sharpes
btstats['maxdd'] = max_drawdowns
btstats['calmar'] = calmars
print(btstats)

# combine stats plots
x_plot = np.array(btstats.index)
fig, axs = plt.subplots(2, 2)
axs[0, 0].bar(x_plot,np.array(btstats['sharpe']), color="blue")
axs[0, 0].set_title('Sharpe Ratio')
axs[0, 1].bar(x_plot, np.array(btstats['sortino']), color="orange")
axs[0, 1].set_title('Sortino Ratio')
axs[1, 0].bar(x_plot, np.array(btstats['maxdd']), color="green")
axs[1, 0].set_title('Max Drawdown')
axs[1, 1].bar(x_plot, np.array(btstats['calmar']), color="red")
axs[1, 1].set_title('Calmar ratio')
plt.show()