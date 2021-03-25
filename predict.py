from urllib.request import urlopen

import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from helper import *

model = get_model_from_az_storage()

# extracts the date and close data and


def get_stock_data_from_ticker(ticker):
    try:
        ticker = str(ticker)
        url = "https://query1.finance.yahoo.com/v7/finance/download/"+ticker + \
            "?period1=1458000000&period2=1915766400&interval=1d&events=history&includeAdjustedClose=True"
        df = pd.read_csv(url, parse_dates=True)
        df = df[['Date', 'Close']]

        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_msft = preprocess_data(df)

        lookback = 100
        x_train, y_train, x_test, y_test = split_data(df_msft, lookback)

        return df, x_test, y_test
    except:
        return "Invalid Ticker"


def evaluate_model(model, x_test, y_test, scaler):
    model.eval()
    y_test_pred = model(x_test)

    # invert predictions
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    return y_test_pred


def predict_stock_price(ticker):
    scaler, x_test, y_test = get_stock_data_from_ticker
    result = evaluate_model(model, x_test, y_test, scaler)
    return result


if __name__ == '__main__':
     predict_stock_price(sys.argv[1])


     
