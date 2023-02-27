import sys
import pandas as pd
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from scipy.stats import norm, t
import matplotlib.pyplot as plt
import yfinance as yf

class VaR:
    def __init__(self, stocks, start, end):
        '''
        Initialize portfolio: Retrieve stock data from Yahoo Finance, calculate returns and mean returns for each stock and the overall portfolio.
        - stocks: list of stock tickers
        - start: start date
        - end: end date
        '''
        # Retrieve stock data from Yahoo Finance
        yf.pdr_override()
        stockData = pdr.get_data_yahoo(stocks, start=start, end=end)
        stockData = stockData['Close']
        # Calculate returns, mean returns, and weights for each stock and the overall portfolio
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        returns = returns.dropna()
        weights = np.random.random(len(returns.columns))
        weights /= np.sum(weights)
        returns['portfolio'] = returns.dot(weights)

        self.returns = returns
        self.meanReturns = meanReturns
        self.covMatrix = covMatrix
        self.weights = weights

    def _portfolioPerformance(self, time):
        meanReturns = self.meanReturns
        covMatrix = self.covMatrix
        weights = self.weights
        returns = np.sum(meanReturns*weights)*time
        std = np.sqrt( np.dot(weights.T, np.dot(covMatrix, weights)) ) * np.sqrt(time)
        return returns, std

    def _historicalVaRPercentile(self, returns, alpha):
        if isinstance(returns, pd.Series):
            return np.percentile(returns, alpha)
        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self._historicalVaRPercentile, alpha=alpha)
        else:
            raise TypeError("Expected returns to be dataframe or series")
        
    def _historicalCVarPercentile(self, returns, alpha):
        if isinstance(returns, pd.Series):
            belowVaR = returns <= self._historicalVaRPercentile(returns, alpha=alpha)
            return returns[belowVaR].mean()
        # A passed user-defined-function will be passed a Series for evaluation.
        elif isinstance(returns, pd.DataFrame):
            return returns.aggregate(self._historicalCVarPercentile, alpha=alpha)
        else:
            raise TypeError("Expected returns to be dataframe or series")
        
    def _monteCarlo(self, sims, time, initialInvestment, plot=True):
        weights = self.weights
        meanReturns = self.meanReturns
        covMatrix = self.covMatrix

        meanM = np.full(shape=(time, len(weights)), fill_value=meanReturns)
        meanM = meanM.T
        #meanM = meanM.time
        portfolio_sims = np.full(shape=(time, sims), fill_value=0.0)
        for m in range(0, sims):
            # MonteCarlo loops
            Z = np.random.normal(size=(time, len(weights)))
            L = np.linalg.cholesky(covMatrix)
            dailyReturns = meanM + np.inner(L, Z)
            portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialInvestment
        
        if plot:
            # Plotting
            plt.plot(portfolio_sims)
            plt.ylabel('Portfolio Value ($)')
            plt.xlabel('Days')
            plt.title('MC simulation of a stock portfolio')
            plt.show()

        return pd.Series(portfolio_sims[-1,:])    

    def historicalVaR(self, alpha, time, initialInvestment):
        """
        Read in a pandas series of returns or a pandas dataframe of returns.
        Output the percentile of the distribution at the given alpha confidence level and dollar VAR,
        given no assumptions about the distribution of returns.
        - alpha: confidence level
        - time: time period in days
        - initialInvestment: initial investment

        -> return [percentile of the distribution, dollar VAR]
        """
        returns = self.returns
        percentile = -self._historicalVaRPercentile(returns, alpha)*np.sqrt(time)
        return [percentile, round(percentile*initialInvestment,2)]    
     
    def historicalCVaR(self, alpha=5, time=100, initialInvestment=10000):
        """
        Read in a pandas series of returns or a pandas dataframe of returns.
        Output the CVaR of the distribution at the given alpha confidence level, given no assumptions about the distribution of returns.
        - alpha: confidence level
        - time: time period in days
        - initialInvestment: initial investment
        
        -> return (conditional Var percentile, dollar CVaR)
        """
        returns = self.returns
        percentile = -self._historicalCVarPercentile(returns, alpha)*np.sqrt(time)
        return [percentile, round(percentile*initialInvestment,2)]
    
    def var_parametric(self, alpha, time, dof, initialInvestment, distribution='normal'):
        """
        Based on portfolio mean and standard deviation of returns, calculate the parametric VaR and dollar VaR based on a given distribution, alpha, degree of freedom, and time period.
        - distribution: 'normal' or 't-distribution'
        - alpha: confidence level
        - time: time period in days
        - dof: degree of freedom
        - initialInvestment: initial investment
        
        -> return [parametric VAR, dollar parametric VAR]
        """
        portfolioReturns, portfolioStd = self._portfolioPerformance(time)
        # because distribution is symmetric
        if distribution == 'normal':
            VaR = norm.ppf(1-alpha/100)*portfolioStd - portfolioReturns
        elif distribution == 't-distribution':
            nu = dof
            VaR = np.sqrt((nu-2)/nu) * t.ppf(1-alpha/100, nu) * portfolioStd - portfolioReturns
        else:
            raise TypeError("Expected distribution type 'normal'/'t-distribution'")
        return [VaR, round(VaR*initialInvestment,2)]
    
    def cvar_parametric(self, alpha, time, dof, initialInvestment, distribution='normal'):
        """
        Based on portfolio mean and standard deviation of returns, calculate the parametric CVaR and dollar CVaR based on a given distribution, alpha, degree of freedom, and time period.
        - distribution: 'normal' or 't-distribution'
        - alpha: confidence level
        - time: time period in days
        - dof: degree of freedom
        - initialInvestment: initial investment

        -> return [parametric CVAR, dollar parametric CVAR]
        """
        portfolioReturns, portfolioStd = self._portfolioPerformance(time)
        if distribution == 'normal':
            CVaR = (alpha/100)**-1 * norm.pdf(norm.ppf(alpha/100))*portfolioStd - portfolioReturns
        elif distribution == 't-distribution':
            nu = dof
            xanu = t.ppf(alpha/100, nu)
            CVaR = -1/(alpha/100) * (1-nu)**(-1) * (nu-2+xanu**2) * t.pdf(xanu, nu) * portfolioStd - portfolioReturns
        else:
            raise TypeError("Expected distribution type 'normal'/'t-distribution'")
        return [CVaR, round(CVaR*initialInvestment,2)]
    
    def MonteCarloVaR(self, sims, time, alpha, initialInvestment, plot=True):
        """ 
        Runs a Monte Carlo simulation of the portfolio and returns the VaR and dollar VaR.
        - sims: number of simulations
        - time: time period in days
        - alpha: confidence level
        - initialInvestment: initial investment
        - plot: boolean, whether to plot the simulation or not

        -> return [percentile on return distribution to a given confidence level alpha, dollar VAR]
        """
        returns = self._monteCarlo(sims, time, initialInvestment, plot)
        if isinstance(returns, pd.Series):
            percentile = np.percentile(returns, alpha)
            return [percentile, round(initialInvestment - percentile,2)]
        else:
            raise TypeError("Expected a pandas data series.")
        
    def MontyCarloCVaR(self, sims, time, alpha, initialInvestment, plot=True):
        """
        Runs a Monte Carlo simulation of the portfolio and returns the CVaR and dollar CVaR.
        - sims: number of simulations
        - time: time period in days
        - alpha: confidence level
        - initialInvestment: initial investment
        - plot: boolean, whether to plot the simulation or not

        -> return [CVaR or Expected Shortfall to a given confidence level alpha, dollar CVAR]
        """
        returns = self._monteCarlo(sims, time, initialInvestment, plot)
        if isinstance(returns, pd.Series):
            belowVaR = returns <= self.MonteCarloVaR(sims,time,alpha,initialInvestment,plot)[0]
            CVaR = returns[belowVaR].mean()
            return [CVaR, round(initialInvestment - CVaR,2)]
        else:
            raise TypeError("Expected a pandas data series.")

#initialize parameters
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=800)
stocks = ['AAPL', 'META', 'PM', 'NFLX', 'GPRO', '^GSPC']
alpha = 5
CI = 100-alpha
time = 100 #days
sims = 1000
initialInvestment = 10000
dof = 6
plot = True

# init VaR object and apply all methods
var = VaR(stocks, start=startDate, end=endDate)
pfPerf = var._portfolioPerformance(time)
hVaR = var.historicalVaR(alpha, time, initialInvestment)
hCVaR = var.historicalCVaR(alpha, time, initialInvestment)
pVaR_n = var.var_parametric(alpha, time, dof, initialInvestment,distribution='normal')
pVaR_t = var.var_parametric(alpha, time, dof, initialInvestment,distribution='t-distribution')
pCVaR_n = var.cvar_parametric(alpha, time, dof, initialInvestment,distribution='normal')
pCVaR_t = var.cvar_parametric(alpha, time, dof, initialInvestment,distribution='t-distribution')
mcVaR = var.MonteCarloVaR(sims, time, alpha, initialInvestment, plot)
mcCVaR = var.MontyCarloCVaR(sims, time, alpha, initialInvestment, plot=False)

# print results of methods
print(f"""
Portfolio: {stocks}
Start Date: {startDate}
End Date: {endDate}
Alpha: {alpha}
CI: {CI}
Time: {time}
Sims: {sims}
Initial Investment: {initialInvestment}
Degree of Freedom: {dof}

Expected Porfolio Return: {round(pfPerf[0]*100,2)}%

---Non-Parametric VaR & CVaR---
Non-Parametric VaR {CI}th CI: ${hVaR[1]['portfolio']}
Non-Parametric Conditional VaR {CI}th CI: ${hCVaR[1]['portfolio']}

---Parametric VaR & CVaR---
Normal VaR {CI}th CI: ${pVaR_n[1]}
Normal Conditional VaR ${CI}th CI: ${pCVaR_n[1]}
t-dist VaR {CI}th CI: ${pVaR_t[1]}
t-dist Conditional VaR {CI}th CI: ${pCVaR_t[1]}

---Monte Carlo VaR & CVaR---
Monte Carlo VaR {CI}th CI: ${mcVaR[1]}
Monte Carlo Conditional VaR {CI}th CI: ${mcCVaR[1]}

---Compare VaR Methods---
Non-Parametric VaR {CI}th CI: ${hVaR[1]['portfolio']}
Parametric Normal VaR {CI}th CI: ${pVaR_n[1]}
Parametric t-dist VaR {CI}th CI: ${pVaR_t[1]}
Monte Carlo VaR {CI}th CI: ${mcVaR[1]}

---Compare CVaR Methods---
Non-Parametric CVaR {CI}th CI: ${hCVaR[1]['portfolio']}
Parametric Normal CVaR {CI}th CI: ${pCVaR_n[1]}
Parametric t-dist CVaR {CI}th CI: ${pCVaR_t[1]}
Monte Carlo CVaR {CI}th CI: ${mcCVaR[1]}
""")
    