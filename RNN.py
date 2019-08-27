import quandl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = quandl.get("BSE/SPBSN5IP", authtoken="WLyLLRyMUTxjnpF-rXx1")
dataset.reset_index(level=0, inplace=True)

#Making Training Dataset
dataset_train=dataset.iloc[:401,:6].values
dataset_train=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
training_set = dataset_train.iloc[:, 1:5].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 30 timesteps and 1 output
X_train = []
y_train = []
for i in range(30, 401):
    X_train.append(training_set_scaled[i-30:i, :])
    y_train.append(training_set_scaled[i, :])
X_train, y_train = np.array(X_train), np.array(y_train)

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 4)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 100))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 4))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 200, batch_size = 30)

# Part 3 - Making the predictions and visualising the results

# Getting values to be predicted
dataset_test=dataset_train=dataset.iloc[401:,:6].values
dataset_test=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
real_stock_price = dataset_test.iloc[:, 1:5].values

# Getting the predicted stock price of 2019
dataset_train=pd.DataFrame(dataset_train,columns=['Date','Open','High','Low','Close'])
dataset_total = pd.concat((dataset_train, dataset_test), axis = 0)
length = len(dataset_total) - len(dataset_test) - 30
inputs = dataset_total.iloc[len(dataset_total) - len(dataset_test) - 30:,1:5 ].values
#inputs = inputs.reshape(-1,5)
inputs = sc.transform(inputs)
X_test = []
for i in range(30, 60):
    X_test.append(inputs[i-30:i, :])
X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
#OPENING PRICE
plt.plot(real_stock_price[:,0], color = 'red', label = 'Real Opening Stock Price')
plt.plot(predicted_stock_price[:,0], color = 'blue', label = 'Predicted Opening Stock Price')
plt.title('Opening Stock Price')
plt.xlabel('Time')
plt.ylabel('Opening Stock Price')
plt.legend()
plt.show()

#HIGHEST PRICE
plt.plot(real_stock_price[:,1], color = 'red', label = 'Real Highest Stock Price')
plt.plot(predicted_stock_price[:,1], color = 'blue', label = 'Predicted Highest Stock Price')
plt.title('Highest Stock Price')
plt.xlabel('Time')
plt.ylabel('Highest Stock Price')
plt.legend()
plt.show()

#LOWEST PRICE
plt.plot(real_stock_price[:,2], color = 'red', label = 'Real Lowest Stock Price')
plt.plot(predicted_stock_price[:,2], color = 'blue', label = 'Predicted Lowest Stock Price')
plt.title('Lowest Stock Price')
plt.xlabel('Time')
plt.ylabel('Lowest Stock Price')
plt.legend()
plt.show()

#CLOSING PRICE
plt.plot(real_stock_price[:,3], color = 'red', label = 'Real Closing Stock Price')
plt.plot(predicted_stock_price[:,3], color = 'blue', label = 'Predicted Closing Stock Price')
plt.title('Closing Stock Price')
plt.xlabel('Time')
plt.ylabel('Closing Stock Price')
plt.legend()
plt.show()

from sklearn.metrics import r2_score
error=[]
for i in range (4):
    error.append(float(r2_score(real_stock_price[:30,i],predicted_stock_price[:,1])))
    
today=dataset.iloc[-31:-1,1:].values
today=np.reshape(today,(1,30,4))
#today=pd.DataFrame(today,columns=['Date','Open','High','Low','Close'])
today_pred=regressor.predict(today)
today_pred = sc.inverse_transform(today_pred)
