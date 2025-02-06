from .func import GeneralizedBell_MF, Linear_CF, Gaussian_MF
from .layers import FuzzificationLayer, h_FuzzificationLayer, FiringLevelsLayer, h_FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer
from .models import ANFIS, h_ANFIS, AntecedentBlock, h_AntecedentBlock
from .training import SONFIS, alt_SONFIS, Hybrid_learning_algorithm, EarlyStopping, Optimizer_training
from .evaluation import get_measures