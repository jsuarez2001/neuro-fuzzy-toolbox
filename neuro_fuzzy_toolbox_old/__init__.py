from .func import gaussian2, weighted_linear
from .layers import FuzzifyLayer, FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer
from .models import Type3ANFIS
from .training import EarlyStopping, obtain_measures, hybrid_algorithm, SONFIS