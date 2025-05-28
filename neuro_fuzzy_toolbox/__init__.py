from .func import GeneralizedBell_MF, Gaussian_MF, Linear_CF
from .layers import FuzzificationLayer, h_FuzzificationLayer, rule_reduced_FuzzificationLayer, FiringLevelsLayer, h_FiringLevelsLayer, rule_reduced_FiringLevelsLayer, NormalizationLayer, ConsequentLayer, alt_ConsequentLayer, OutputLayer
from .models import ANFIS, h_ANFIS, rule_reduced_ANFIS
from .training import SONFIS, Hybrid_learning_algorithm, Basic_optimizer_training_algorithm, Double_optimizer_training_algorithm, EarlyStopping