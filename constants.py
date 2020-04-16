import os
import models

# Some constants paths
DIR_BASE = os.path.abspath(os.path.dirname(__file__))
DATA = os.path.join(DIR_BASE, 'data')
MODELS = os.path.join(DIR_BASE, 'models')

# To be modified if you want to load or train the model, if True, will train a model
TRAIN_standard_model = False
TRAIN_robust_model = False
TRAIN_large_robust_model = False

# Paths of already trained model, to be modified as pleased
STANDARD_trained_model = os.path.join(MODELS, 'standard_model_78acc.h5')
ROBUST_trained_model = os.path.join(MODELS, 'Attacked_model_standard_robust2020-04-15.h5')
LARGE_ROBUST_trained_model = os.path.join(MODELS, 'Attacked_model_large_robust2020-04-15.h5')

# Attacks configuration
MAKE_ATTACK = False         # Will load attacked data if False
attack_style = 'PGD'
attack_delta = 0.03
attack_nb_iter = 3

# Paths of already attacked data
ATTACKED_TRAIN = os.path.join(DATA, 'attack_train_PGD.npy')
ATTACKED_TEST = os.path.join(DATA, 'attack_test_PGD.npy')

# Configuration of models (to be modified as pleased)
config_standard_model = models.ModelConfig(
            conv_layers=[(256, 3), (256, 3), (256, 3)],
            epochs=100,
            batch_size=128,
            validation_split=0.1,
            nb_neurons_on_1st_FC_layer=32,
            lr=0.01
        )

config_robust_model = models.ModelConfig(
            conv_layers=[(256, 3), (256, 3), (256, 3)],
            epochs=100,
            batch_size=128,
            validation_split=0.1,
            nb_neurons_on_1st_FC_layer=32,
            lr=0.01
        )

config_large_robust_model = models.ModelConfig(
            conv_layers=[(256, 3), (256, 3), (256, 3)],
            epochs=100,
            batch_size=128,
            validation_split=0.1,
            nb_neurons_on_1st_FC_layer=32,
            lr=0.01
        )

# constants labels of CIFAR10 to make some visualization
class_to_name = {
    0:	"plane",
    1:	"car",
    2:	"bird",
    3:	"cat",
    4:	"deer",
    5:	"dog",
    6:	"frog",
    7:	"horse",
    8:	"ship",
    9:	"truck"
}
