
import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = quandl.get("BSE/SPBSN5IP", authtoken="WLyLLRyMUTxjnpF-rXx1")
dataset.reset_index(level=0, inplace=True)

#Making Training Dataset
dataset_train=dataset.iloc[:-30,:6].values
dataset_train=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
training_set = dataset_train.iloc[:, 1:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
c = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

length=len(dataset)

# Creating a data structure with 30 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, length-30):
    X_train.append(training_set_scaled[i-30:i, :])
    y_train.append(training_set_scaled[i, :])
X_train, y_train = np.array(X_train), np.array(y_train)

#Fitting
#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 100)
#regressor.fit(X_train[:,:,0], y_train[:,0])

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf', gamma = 0.1)
regressor.fit(X_train[:,:,0], y_train[:,0])


# Getting values to be predicted
dataset_test=dataset_train=dataset.iloc[-30:,:6].values
dataset_test=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
real_stock_price = dataset_test.iloc[:, 1:5].values

# Getting the predicted stock price of 2019
dataset_train=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
length = len(dataset_total) - len(dataset_test) - 30
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - 30:,1:5 ].values
inputs = sc.transform(inputs)
X_test = []
for i in range(30, 60):
    X_test.append(inputs[i-30:i, :])
X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test[:,:,0])
make=np.zeros((30,4))
predicted_stock_price =np.reshape(predicted_stock_price,(30,1))
predicted_stock_price=predicted_stock_price+make
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


from sklearn.metrics import r2_score
error = r2_score(real_stock_price[:30,1],predicted_stock_price[:,1])

# Visualising the results
#OPENING PRICE
plt.plot(real_stock_price[:,0], color = 'red', label = 'Real Opening Stock Price')
plt.plot(predicted_stock_price[:,0], color = 'blue', label = 'Predicted Opening Stock Price')
plt.title('Opening Stock Price')
plt.xlabel('Time')
plt.ylabel('Opening Stock Price')
plt.legend()
plt.show()