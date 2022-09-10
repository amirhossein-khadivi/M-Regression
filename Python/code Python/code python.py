# Multiple Linear Regression

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics import gofplots as sgg
import scipy.stats as ss


data = pd.DataFrame(
    {
       'y' : [1.45,1.93,0.81,0.61,1.55,0.95,0.45,1.14,0.74,0.98,1.41,0.81,0.89
              ,0.68,1.39,1.53,0.91,1.49,1.38,1.73,1.11 ,1.68,0.66,0.69,1.98],
       'x1' : [0.58,0.86,0.29,0.20,0.56,0.28,0.08,0.41,0.22,0.35,0.59,0.22
              ,0.26,0.12,0.65,0.70,0.30,0.70,0.39,0.72,0.45,0.81,0.04,0.20,0.95],
       'x2' : [0.71,0.13,0.79,0.20,0.56,0.92,0.01,0.60,0.70,0.73,0.13,
              0.96,0.27,0.21,0.88,0.30,0.15,0.09,0.17,0.25,0.30,0.32,0.82,0.98,0.00]
    }
    , dtype = float
)
print(data)


y = np.array(data['y']).reshape(25,1)
x = np.array(data[['x1','x2']]).reshape(25,2)


xc = sm.add_constant(x)
lm1 = sm.OLS(data['y'] , xc).fit()
lms1 = lm1.summary()
print(lms1)

plt.scatter(x = data['x1'] , y = data['x2'] , color = 'blue')
plt.show()

z = sm.graphics.plot_partregress_grid(lm1)
z.tight_layout(pad=1.0)

xc1 = sm.add_constant(x[:,0])
lm2 = sm.OLS(data['y'] , xc1).fit()
lms2 = lm2.summary()
print(lms2)

fit_value = lm2.predict(xc1)
fit_value

resi = lm2.resid
data1 = pd.DataFrame({'resi': resi , 'fit_value': fit_value})
data1


sgg.qqplot(data1['resi'] , line = 's')
plt.show()


a = ss.shapiro(data1['resi'])
b = ss.normaltest(data1['resi'])
c = ss.anderson(data1['resi'])

print(a) ; print(b)

print('Anderson.result','  statistic : ',c.statistic) ; cpp = c.statistic
for i in range(len(c.critical_values)):
    q = c.significance_level[i] ; w = c.critical_values[i]
    print(q,' : ',w)
    if q == 5 :
        cp = w

plt.scatter(x = data1['fit_value'] , y = data1['resi'] , color = 'red')
plt.axhline(0)
plt.axhline(0.3 , color = 'yellow')
plt.axhline(-0.3 , color = 'yellow')
plt.show()

sm.graphics.tsa.plot_acf(data1['resi'] , color = 'red')
plt.show()

