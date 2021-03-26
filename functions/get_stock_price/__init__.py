import logging
from predict import predict_stock_price

import azure.functions as func


from predict import predict_stock_price


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # GET request or URL www.function.com?ticker=msft
    ticker = req.params.get('ticker')

    results = predict_stock_price(ticker)

    if not ticker:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            # POST request with { "ticker": "msft" }
            ticker = req_body.get('ticker')

    if ticker:
        return func.HttpResponse(f'{results}')
    else:
        return func.HttpResponse(
            "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
            status_code=200
        )
