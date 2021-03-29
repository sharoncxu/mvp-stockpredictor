import requests
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import io
from flask import Flask, render_template, request, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

@app.route('/')
def index():
    return 'hello world'

@app.route("/ticker/", methods=['POST'])
def ticker():
    ticker = request.form.get('Ticker')
    ticker = ticker.upper()
    url = f"https://mvp-stockpredictor.azurewebsites.net/api/get_stock_price?ticker={ticker}&code=WvXjCoODgLc2HAWd8tF/rt/7hs/6Quqk5aXX2qRtlgSUmi3iNkyFWQ=="
    content = requests.get(url).content
    content = content.decode('utf-8')
    df_list = ast.literal_eval(content)

    img = io.BytesIO()

    loc = plticker.MultipleLocator(base=25)

    df = pd.DataFrame(df_list, columns=["Date", "Actual", "Predicted"])

    figure, axes = plt.subplots(figsize=(16, 7))
    dates = df["Date"].to_numpy().reshape(-1)

    axes.plot(dates, df["Actual"], color='red', label=f'Real {ticker} Stock Price')
    axes.plot(dates, df["Predicted"], color='blue',
            label=f'Predicted {ticker} Stock Price')
    axes.xaxis.set_major_locator(loc)

    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{ticker} Stock Price')
    plt.legend()

    output = io.BytesIO()
    FigureCanvas(figure).print_png(output)
    
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/form/')
def form():
    return render_template('form.html')
