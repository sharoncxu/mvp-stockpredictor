import logging
import azure.functions as func
from ..predict import predict_stock_price

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # GET request or URL www.function.com?ticker=msft
    ticker = req.params.get('ticker')

    if ticker:
        results = predict_stock_price(ticker)
        return func.HttpResponse(f'{results}')
    else:
        return func.HttpResponse(
            "Hello MVPs! We're happy to have you :)",
            status_code=200
        )
