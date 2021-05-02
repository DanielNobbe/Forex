'''
All possible trades
'''

# TODO: There is a better way to do this with generating only what you need, but this is alse nice to have because of the info, but it is slower, since you need to filter all the useless stuf out
# TODO: Check FixedPriceOrder
    
def filter_dict(Dictionary):
    To_be_deleted = []
    for key, value in Dictionary.items():
        if isinstance(value, dict):
            Dictionary[key] = FilterDict(Dictionary[key])
            if not Dictionary[key]:
                To_be_deleted.append(key)
        if value == "":
            To_be_deleted.append(key)   
    for key in To_be_deleted:
        del Dictionary[key]
    return Dictionary

TimeInForce = {
    "GTC": "The Order is “Good unTil Cancelled”",
    "GTD": "The Order is “Good unTil Date” and will be cancelled at the provided time",
    "GFD": "The Order is “Good For Day” and will be cancelled at 5pm New York time",
    "FOK": "The Order must be immediately “Filled Or Killed”",
    "IOC": "The Order must be “Immediately partially filled Or Cancelled”"
    }

PositionFill = {
    "OPEN_ONLY":    "When the Order is filled, only allow Positions to be opened or extended.",
    "REDUCE_FIRST": "When the Order is filled, always fully reduce an existing Position before opening a new Position.",
    "REDUCE_ONLY": 	"When the Order is filled, only reduce an existing Position.",
    "DEFAULT": 	    "When the Order is filled, use REDUCE_FIRST behaviour for non-client hedging Accounts, and OPEN_ONLY behaviour for client hedging Accounts."
    }

TriggerCondition = {
    "DEFAULT":	 "Trigger an Order the “natural” way: compare its price to the ask for long Orders and bid for short Orders.",
    "INVERSE":	 "Trigger an Order the opposite of the “natural” way: compare its price the bid for long Orders and ask for short Orders.",
    "BID":	     "Trigger an Order by comparing its price to the bid regardless of whether it is long or short.",
    "ASK":	     "Trigger an Order by comparing its price to the ask regardless of whether it is long or short.",
    "MID":	     "Trigger an Order by comparing its price to the midpoint regardless of whether it is long or short." 
    }

FixedPriceOrder = { # Check if this is right
        "order": {
            "type":                     "FIXED_PRICE"           , # required
            "instrument":               ""                      , # required
            "units":                    ""                      , # required  
            "timeInForce":              "GTC"                   , # required
            "priceBound":               ""                      , 
            "positionFill":             "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                               ,
            "takeProfitOnFill": {
                "price":                ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "stopLossOnFill": {                                   # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }                                           ,
                "guaranteed":           ""                        #         , Will be removed in a future update
                }                                               ,
            "guaranteedStopLossOnFill": {                         # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "trailingStopLossOnFill": {
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "tradeClientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                
            }
        }

MarketOrder = {
        "order": {
            "type":                     "MARKET"                , # required
            "instrument":               ""                      , # required
            "units":                    ""                      , # required  
            "timeInForce":              "GTC"                   , # required
            "priceBound":               ""                      , 
            "positionFill":             "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                               ,
            "takeProfitOnFill": {
                "price":                ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "stopLossOnFill": {                                   # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }                                           ,
                "guaranteed":           ""                        #         , Will be removed in a future update
                }                                               ,
            "guaranteedStopLossOnFill": {                         # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "trailingStopLossOnFill": {
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "tradeClientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

LimitOrder = {
        "order": {
            "type":                     "LIMIT"                 , # required
            "instrument":               ""                      , # required
            "units":                    ""                      , # required  
            "price":                    ""                      , # required
            "timeInForce":              "GTC"                   , # required
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "priceBound":               ""                      , 
            "positionFill":             "DEFAULT"               , # required
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                               ,
            "takeProfitOnFill": {
                "price":                ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "stopLossOnFill": {                                   # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }                                           ,
                "guaranteed":           ""                        #         , Will be removed in a future update
                }                                               ,
            "guaranteedStopLossOnFill": {                         # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "trailingStopLossOnFill": {
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "tradeClientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

StopOrder = {
        "order": {
            "type":                     "STOP"                  , # required
            "instrument":               ""                      , # required
            "units":                    ""                      , # required  
            "price":                    ""                      , # required
            "timeInForce":              "GTC"                   , # required
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "priceBound":               ""                      , 
            "positionFill":             "DEFAULT"               , # required
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                               ,
            "takeProfitOnFill": {
                "price":                ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "stopLossOnFill": {
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }                                           ,
                "guaranteed":           ""                        #         , Will be removed in a future update
                }                                               ,
            "guaranteedStopLossOnFill": {
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "trailingStopLossOnFill": {
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "tradeClientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

MarketIfTouchedOrder = {
        "order": {
            "type":                     "MARKET_IF_TOUCHED"     , # required
            "instrument":               ""                      , # required
            "units":                    ""                      , # required  
            "price":                    ""                      , # required
            "timeInForce":              "GTC"                   , # required
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "priceBound":               ""                      , 
            "positionFill":             "DEFAULT"               , # required
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }                                               ,
            "takeProfitOnFill": {
                "price":                ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "stopLossOnFill": {                                   # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }                                           ,
                "guaranteed":           ""                        #         , Will be removed in a future update
                }                                               ,
            "guaranteedStopLossOnFill": {                         # Either of price and distance is needed
                "price":                ""                      ,
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "trailingStopLossOnFill": {
                "distance":             ""                      ,
                "timeInForce":          ""                      , #         , GTC, GTD or GFD
                "gtdTime":              ""                      , #         , If timeInForce = GTD
                "clientExtensions": {
                    "id":               ""                      ,
                    "tag":              ""                      ,
                    "comment":          ""                      
                    }
                }                                               ,
            "tradeClientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

TakeProfitOrder = {
        "order": {
            "type":                     "TAKE_PROFIT"           , # required
            "tradeID":                  ""                      , # required
            "clientTradeID":            ""                      ,
            "price":                    ""                      , # required
            "timeInForce":              "GTC"                   , # required, GTC, GTD or GFD
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

StopLossOrder = {
        "order": {
            "type":                     "STOP_LOSS"             , # required
            "tradeID":                  ""                      , # required
            "clientTradeID":            ""                      ,
            "price":                    ""                      , # required
            "distance":                 ""                      ,
            "timeInForce":              "GTC"                   , # required, GTC, GTD or GFD
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "triggerCondition":         "DEFAULT"               , # required
            "guarenteed":               ""                      , #         , Will be removed in a future update
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

GuarenteedStopLossOrder = {
        "order": {
            "type":                     "GUARANTEED_STOP_LOSS"  , # required
            "tradeID":                  ""                      , # required
            "clientTradeID":            ""                      ,
            "price":                    ""                      , # required
            "distance":                 ""                      ,
            "timeInForce":              "GTC"                   , # required, GTC, GTD or GFD
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }

TrailingStopLossOrder = {
        "order": {
            "type":                     "TRAILING_STOP_LOSS"    , # required
            "tradeID":                  ""                      , # required
            "clientTradeID":            ""                      ,
            "price":                    ""                      , # required
            "distance":                 ""                      ,
            "timeInForce":              "GTC"                   , # required, GTC, GTD or GFD
            "gtdTime":                  ""                      , #         , If timeInForce = GTD
            "triggerCondition":         "DEFAULT"               , # required
            "clientExtensions": {
                "id":                   ""                      ,
                "tag":                  ""                      ,
                "comment":              ""                      
                }
            }
        }