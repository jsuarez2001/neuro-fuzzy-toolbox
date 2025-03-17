from .func import GeneralizedBell_MF, Gaussian_MF, Linear_CF
from .layers import FuzzificationLayer, h_FuzzificationLayer, FiringLevelsLayer, h_FiringLevelsLayer, rule_reduced_FuzzificationLayer, NormalizationLayer, ConsequentLayer, alt_ConsequentLayer, OutputLayer
from .models import ANFIS, h_ANFIS, rule_reduced_ANFIS
from .training import SONFIS, alt_SONFIS, Hybrid_learning_algorithm, EarlyStopping, Basic_optimizer_training_algorithm, Double_optimizer_training_algorithm
from .evaluation import get_measures