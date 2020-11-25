#!/usr/bin/env python3
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats

my_pandas_file = pd.read_csv('cs_24.csv')
y_data = my_pandas_file.get('voltage_integral')

x_data = np.arange(0,len(y_data), 1)
x_data_composed = x_data.reshape(-1,1)
correlation = pearsonr(x_data, y_data)
print(correlation)


lin_regression = LinearRegression().fit(x_data_composed, y_data)

model_line = lin_regression.predict(x_data_composed)


plt.plot(y_data, label='data')
plt.plot(model_line,label='model')
plt.xlabel('Cilos')
plt.ylabel('volts x seconds')
plt.title('Voltage integral CCCT charge during batery life')
plt.legend()
plt.ylim(0,8000)
plt.show()
#plt.savefig('data.jpg', format='jpg')



plt.close()


def get_prediction_interval(prediction, y_test, test_predictions, pi=.95):
    '''
    Get a prediction interval for a linear regression.
    
    INPUTS: 
        - Single prediction, 
        - y_test
        - All test set predictions,
        - Prediction interval threshold (default = .95) 
    OUTPUT: 
        - Prediction interval for single prediction
    '''
    
    #get standard deviation of y_test
    sum_errs = np.sum((y_test - test_predictions)**2)
    stdev = np.sqrt(1 / (len(y_test) - 2) * sum_errs)
#get interval from standard deviation
    one_minus_pi = 1 - pi
    ppf_lookup = 1 - (one_minus_pi / 2)
    z_score = stats.norm.ppf(ppf_lookup)
    interval = z_score * stdev
    
    
#generate prediction interval lower and upper bound cs_24
    lower, upper = prediction - interval, prediction + interval
    return lower, prediction, upper


## Plot and save confidence interval of linear regression  - 95% cs_24
lower_vet = []
upper_vet = []

for i in model_line:
    lower, prediction, upper =  get_prediction_interval(i, y_data, model_line)
    lower_vet.append(lower)
    upper_vet.append(upper)
    
plt.fill_between(np.arange(0,len(y_data),1),upper_vet, lower_vet, color='b',label='Confidence Interval')
plt.plot(np.arange(0,len(y_data),1),y_data,color='orange',label='Real data')
plt.plot(model_line,'k',label='Linear regression')
plt.xlabel('Ciclos')
plt.ylabel('Volts x seconds')
plt.title('95% confidence interval')
plt.legend()
plt.ylim(-1000,8000)
plt.savefig('confid_int_cs24.eps', format='eps')
plt.show()