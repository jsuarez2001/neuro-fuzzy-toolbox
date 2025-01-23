from .func import GeneralizedBell_MF, Linear_CF, Gaussian_MF
from .layers import FuzzificationLayer, FiringLevelsLayer, NormalizationLayer, ConsequentLayer, OutputLayer
from .models import ANFIS
from .training import SONFIS, exp_SONFIS, alt_SONFIS, Hybrid_learning_algorithm, EarlyStopping, classical_consequents_estimation_with_OLS
from .evaluation import get_measures