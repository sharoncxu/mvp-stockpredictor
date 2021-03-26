import requests
from flask import Flask, render_template, request
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello, World!"

@app.route("/ticker/", methods = ['POST'])
def ticker():
    ticker = request.form.get('Ticker')
    url = f"https://mvp-stockpredictor.azurewebsites.net/api/get_stock_price?ticker={ticker}&code=WvXjCoODgLc2HAWd8tF/rt/7hs/6Quqk5aXX2qRtlgSUmi3iNkyFWQ=="
    content = requests.get(url).content
    print(content)
    print(url)
    return render_template('data.html', content = content)

@app.route('/form/')
def form():
    return render_template('form.html')
