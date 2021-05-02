from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.startswith('_')]

from .markov_kernel import MarkovKernel
from .RNN import CandleLSTM

ARCHITECTURES = {
    "markov_kernel":    MarkovKernel,
    "lstm":             CandleLSTM,
}