"""MC1-P2: Optimize a portfolio.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		   	  			    		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative  		   	  			    		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.

We do grant permission to share solutions privately with non-students such  		   	  			    		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)  		   	  			    		  		  		    	 		 		   		 		  
GT User ID: tb34 (replace with your User ID)  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 900897987 (replace with your GT ID)  		   	  			    		  		  		    	 		 		   		 		  
"""
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
import scipy.optimize as spo


def portfolio_value(prices, allocations):
    normed = prices / prices.iloc[0]
    alloced = normed*allocations
    pos_value = alloced * 1
    return pos_value.sum(axis=1)


def daily_returns(df):
    daily_return = (df / df.shift(1)) - 1
    daily_return.iloc[0] = 0
    return daily_return


def sharpe(daily_returns):
    s = math.sqrt(252)*(daily_returns.mean()/daily_returns.std())
    return s


def minimize(allocations, prices):
    port_val = portfolio_value(prices, allocations)
    daily_return = daily_returns(port_val)
    s = sharpe(daily_return)
    return -s


# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def optimize_portfolio(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,1,1),
                       syms=['GOOG','AAPL','GLD','XOM'], gen_plot=False):

    # Read in adjusted closing prices for given symbols, date range  		   	  			    		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices_all = prices_all / prices_all.iloc[0]
    prices = prices_all[syms]  # only portfolio symbols
    prices_spy = prices_all['SPY']  # only SPY, for comparison later

    # find the allocations for the optimal portfolio  		   	  			    		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case  		   	  			    		  		  		    	 		 		   		 		  
    # allocs = np.asarray([0.35, 0.35, 0.10, 0.10, 0.10]) # add code here to find the allocations
    allocs = spo.minimize(minimize, np.random.dirichlet(np.ones(prices.shape[1])), args=(prices,), method="SLSQP",
                          bounds=((0,1),)*prices.shape[1],
                          constraints=(
                              {'type':'eq', 'fun': lambda x:  x.sum()-1}
                          ))["x"]

    # Get daily portfolio value
    # port_val = prices_SPY  # add code here to compute daily portfolio values
    port_val = portfolio_value(prices, allocs)

    daily_return = daily_returns(port_val)
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats
    cr = (port_val[-1]/port_val[0])-1
    adr = daily_return.mean()
    sddr = daily_return.std()
    sr = sharpe(daily_return)

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:  		   	  			    		  		  		    	 		 		   		 		  
        # add code to plot here  		   	  			    		  		  		    	 		 		   		 		  
        df_temp = pd.concat([port_val, prices_spy], keys=['Portfolio', 'SPY'], axis=1)
        plot_data(df_temp)

    return allocs, cr, adr, sddr, sr


def test_code():
    # This function WILL NOT be called by the auto grader  		   	  			    		  		  		    	 		 		   		 		  
    # Do not assume that any variables defined here are available to your function/code  		   	  			    		  		  		    	 		 		   		 		  
    # It is only here to help you set up and test your code  		   	  			    		  		  		    	 		 		   		 		  

    # Define input parameters  		   	  			    		  		  		    	 		 		   		 		  
    # Note that ALL of these values will be set to different values by  		   	  			    		  		  		    	 		 		   		 		  
    # the autograder!  		   	  			    		  		  		    	 		 		   		 		  

    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2009, 1, 1)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'IBM']

    # Assess the portfolio
    allocations, cr, adr, sddr, sr = optimize_portfolio(sd=start_date, ed=end_date,
                                                        syms=symbols, gen_plot=True)

    # Print statistics  		   	  			    		  		  		    	 		 		   		 		  
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Symbols:", symbols)
    print("Allocations:", allocations)
    print("Sharpe Ratio:", sr)
    print("Volatility (stdev of daily returns):", sddr)
    print("Average Daily Return:", adr)
    print("Cumulative Return:", cr)

if __name__ == "__main__":
    # This code WILL NOT be called by the auto grader
    # Do not assume that it will be called  		   	  			    		  		  		    	 		 		   		 		  
    test_code()  		   	  			    		  		  		    	 		 		   		 		  
