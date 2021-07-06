"""
Contains all granularities that the OandaV20 API natively uses,
and maps them to seconds for internal use for us.
"""
gran_to_sec = {
    "S5": 5,
    "S10": 10,
    "S15": 15,
    "S30": 30,
    "M1": 60,
    "M2": 120,
    "M4": 240,
    "M5": 300,
    "M10": 600,
    "M15": 900,
    "M30": 1800,
    "H1": 3600,
    "H2": 7200,
    "H3": 10800,
    "H4": 14400,
    "H6": 21600,
    "H8": 28800,
    "H12": 43200,
    "D": 86400,
    "W": 604800,
    "D30": 2592000, # Months do not always have the same number of seconds
    # "M": "1 month candlesticks, aligned to first day of the month",
}

sec_to_gran = {value: key for key, value in gran_to_sec.items()}
