import tushare as ts
import pandas as pd
import statsmodels.api as sm
data = ts.get_k_data('600858', start='2016-02-05',end='2016-06-05')
share_change=data['close']-data['open']
share_change.index=pd.Index(sm.tsa.datetools.dates_from_range('1','78'))
from statsmodels.tsa.stattools import adfuller #检测序列平稳性，单位根检验。
dftest = adfuller(share_change, autolag='AIC')
print(dftest[1])#在保证ADF检验的p<0.01的情况下，阶数越小越好
from statsmodels.stats.diagnostic import acorr_ljungbox
p_value = acorr_ljungbox(share_change, lags=1)#检测autocorrelation,P>0.5代表有相关
share_acf = sm.graphics.tsa.plot_acf(share_change,lags=40)# acf图
share_pacf = sm.graphics.tsa.plot_pacf(share_change,lags=40)#pacf图
print(share_acf,share_pacf)
share_change1=list(share_change)
arma_mod1 = sm.tsa.ARMA(share_change1,(2,1)).fit()
print(arma_mod1.aic,arma_mod1.bic,arma_mod1.hqic)
arma_mod2 = sm.tsa.ARMA(share_change1,(2,2)).fit()
print(arma_mod2.aic,arma_mod2.bic,arma_mod2.hqic)
arma_mod3 = sm.tsa.ARMA(share_change1,(1,0)).fit()
print(arma_mod3.aic,arma_mod3.bic,arma_mod3.hqic)
arma_mod4 = sm.tsa.ARMA(share_change1,(0,1)).fit()
print(arma_mod4.aic,arma_mod4.bic,arma_mod4.hqic)
resid = arma_mod4.resid
from statsmodels.graphics.api import qqplot
figqq = qqplot(resid)
print(figqq)
predict_ts = arma_mod2.predict(start=79,end=82)
