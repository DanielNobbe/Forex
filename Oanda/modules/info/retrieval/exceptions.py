"""
Contains all custom exceptions for retrieval.
"""

class MarketClosedError(Exception):
    """
    Specific error that is raised when attempting to trade while market
    is closed.
    """

    def __init__(self, message=''):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Could not retrieve enough candlesticks to make a prediction, "\
            "but it seems this is not due to a too tight soft margin. "\
                "Probably the markets are closed. {self.message}"

class MissingSamplesError(Exception):
    """
    Error raised when retrieval could not find enough samples to make
    a prediction. This is raised when it is not conclusive that the markets
    are closed.
    """

    def __init__(self, message=''):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"Could not retrieve enough candlesticks to make a prediction. " \
        "This could be due to the markets being closed. Try to allow (a larger) "\
            "soft margin for retrieval. {self.message}"
  