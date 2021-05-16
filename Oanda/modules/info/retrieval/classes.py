from sortedcontainers import SortedDict

class HistoryArgs():
    """
    Class to hold arguments for retrieval of historical data.
    TODO: Add checks to this
    TODO: Use NameSpace object for this if we are doing no checks
    TODO: Add __slots__ to these objects to decrease object size
    """
    def __init__(self):
        self.instrument = None
        self.start_time = None
        self.granularity = None
        self.max_count = None

class CandleStickValues():
    """
    Class to hold the values for a series of candlesticks, in lists of
    opens, highs, lows and closes.
    Can only be initialised using the from_lists or empty classmethods.
    TODO: Convert this directly to tensors
    TODO: Remove unnecessary information
    TODO: convert candlestickvalues and instrumentseries data handling to general method that
        uses dicts to init the __dict__ of objects  
    TODO: Add __slots__ to these objects to decrease object size  
    """

    __create_key = object()

    def __init__(self, create_key, open, high, low, close):
        """
        Initialises a CandleStickValues object. Should not be called 
        directly, but through the from_lists or empty classmethods.
        Asks for a create_key to monitor this, since the key cannot
        be accessed from outside this class.
        """
        assert (create_key == CandleStickValues.__create_key), "Don't use the normal init function!"
        self.opens = open
        self.highs = high
        self.lows = low
        self.closes = close
    
    @classmethod
    def from_lists(cls, open, high, low, close):
        """
        Initialise a CandleStickValues instance using lists of opens,
        highs, lows and closes.
        """
        assert (isinstance(open, list) and isinstance(high, list) 
            and isinstance(low, list) and isinstance(close, list)), "Initilization inputs for CandleStickValues object must all be lists"
        return CandleStickValues(cls.__create_key, open, high, low, close)
    
    @classmethod
    def empty(cls):
        """
        Initialise an empty CandleStickValues instance, to which entries
        can be added.
        """
        return CandleStickValues(cls.__create_key, [], [], [], [])
    
    def append(self, open, high, low, close):
        """
        Append values for a single candlestick. Calls list.append for 
        each value list.
        """
        self.opens.append(open)
        self.highs.append(high)
        self.lows.append(low)
        self.closes.append(close)
    
    def extend(self, opens, highs, lows, closes):
        """
        Extend with values for multiple candlesticks. Uses list.extend
        for each value list.
        """
        self.opens.extend(opens)
        self.highs.extend(highs)
        self.lows.extend(lows)
        self.closes.extend(closes)
    
class InstrumentSeries():
    """
    Contains a time series of Candlesticks for a specific instrument,
    such as the currency pair EUR_USD. Keeps track of the values of the
    candlesticks at each point in time in a CandleStickValues object,
    and stores the metadata: timestamps, volumes and 'complete' (whether
    the current candlestick is completed or still ongoing).
    Can only be initialised using the from_lists or empty class-methods.
    TODO: Add granularity to instrument series object (and granularity check on extension)
    TODO: convert candlestickvalues and instrumentseries data handling to general method that
        uses dicts to init the __dict__ of objects
    TODO: Add __slots__ to these objects to decrease object size
    """
    def __init__(self):
        """
        Defined to raise an error when used. 
        """
        raise NotImplementedError("Please initialise this object using "
        "the from_lists or empty methods.")
        # Defining values here so pylint does not complain.
        self.values = None
        self.times = None
        self.volumes = None
        self.completes = None
        self.timedict = None

    @classmethod
    def from_lists(cls, opens, highs, lows, closes, times, volumes, completes):
        """
        Define a InstrumentSeries object using lists of values, time-
        stamps and other metadata. Corresponds directly to the values
        returned by the Oanda API.
        """
        assert(
                isinstance(opens, list) 
            and isinstance(highs, list) 
            and isinstance(lows, list) 
            and isinstance(closes, list)
            and isinstance(times, list) 
            and isinstance(volumes, list)
            and isinstance(completes, list)
        ), "Initilization inputs for InstrumentSeries object must all be lists"


        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.from_lists(opens, highs, lows, closes)
        obj.times = times
        obj.volumes = volumes
        obj.completes = completes
        obj.timedict = SortedDict({time: value for time, value in zip(times,[tupl for tupl in zip(opens,highs,lows,closes)])})
        #TODO: modify this to contain an object for each candlestick 
        return obj
    
    @classmethod
    def empty(cls):
        """
        Define an empty InstrumentSeries object, to which entries can be
        added through the append and extend methods.
        """
        obj = cls.__new__(cls)
        super(InstrumentSeries, obj).__init__()
        obj.values = CandleStickValues.empty()
        obj.times = []
        obj.volumes = []
        obj.completes = []
        obj.timedict = SortedDict()
        return obj

    def append(self, open, high, low, close, time, volume, complete):
        """
        Append a single candlestick. Uses list.append internally.
        """
        self.values.append(open, high, low, close)
        self.times.append(time)
        self.volumes.append(volume)
        self.completes.append(complete)
        self.timedict[time] = (open, high, low, close)

    def extend(self, opens, highs, lows, closes, times, volumes, completes):
        """
        Extend with multiple candlesticks. Uses list.extend internally.
        """
        self.values.extend(opens, highs, lows, closes)
        self.times.extend(times)
        self.volumes.extend(volumes)
        self.completes.extend(completes)
        self.timedict.update( {time: value for time, value in zip(times,[tupl for tupl in zip(opens,highs,lows,closes)])} )
    
    def __len__(self):
        return len(self.times)