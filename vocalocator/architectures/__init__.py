from .base import VocalocatorArchitecture
from .ensemble import VocalocatorEnsemble
from .simplenet import VocalocatorSimpleNetwork
from .simplenet_crosscorr import CorrSimpleNetwork
from .wavenet_arch import Wavenet

__all__ = [
    VocalocatorSimpleNetwork,
    VocalocatorEnsemble,
    CorrSimpleNetwork,
    Wavenet,
]
