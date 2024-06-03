from .base import GerbilizerArchitecture
from .ensemble import GerbilizerEnsemble
from .simplenet import GerbilizerSimpleNetwork
from .simplenet_crosscorr import CorrSimpleNetwork
from .wavenet_arch import Wavenet

__all__ = [
    GerbilizerSimpleNetwork,
    GerbilizerEnsemble,
    CorrSimpleNetwork,
    Wavenet,
]
