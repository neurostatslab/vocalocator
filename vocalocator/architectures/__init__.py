from .base import VocalocatorArchitecture
from .conformer import ResnetConformer
from .ensemble import VocalocatorEnsemble
from .simplenet import VocalocatorSimpleNetwork
from .simplenet_crosscorr import CorrSimpleNetwork
from .wavenet_arch import Wavenet

__all__ = [
    VocalocatorSimpleNetwork,
    VocalocatorEnsemble,
    CorrSimpleNetwork,
    Wavenet,
    ResnetConformer,
]
