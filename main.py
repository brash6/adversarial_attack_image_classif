import os
from tensorflow.keras.models import load_model
import numpy as np

import constants as cst
import import_data
import models
import attack_function
import visualization

if __name__ == '__main__':
    # Load CIFAR10 data
    x_train, y_train, x_test, y_test = import_data.format_data()

    # Train or Load standard model (trained on not attacked data)
    if cst.TRAIN_standard_model:
        print("Training a new standard model")
        model = models.train_model(x_train, y_train, cst.config, 'standard_model')
    else:
        print("Loading standard model")
        model = load_model(cst.STANDARD_trained_model)

    # Attack Data
    if cst.MAKE_ATTACK:
        print("Attacking data")
        x_train_attacked = attack_function.make_attack(x_train, y_train, cst.attack_delta, cst.attack_nb_iter, model,
                                                       'x_train_attacked', cst.attack_style)
        x_test_attacked = attack_function.make_attack(x_test, y_test, cst.attack_delta, cst.attack_nb_iter, model,
                                                      'x_test_attacked', cst.attack_style)
    else:
        x_train_attacked = np.load(cst.ATTACKED_TRAIN)
        x_test_attacked = np.load(cst.ATTACKED_TEST)

    # Show the effect of the attack on model predictions
    print("On not attacked data, on the test data the model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test, y_test, model)
    print("On attacked data, on the test data the model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test_attacked, y_test, model)

    # Train or Load a robust model but only on attacked data
    if cst.TRAIN_robust_model:
        print("Training a new only attack robust model")
        robust_model = models.train_model(x_train_attacked, y_train, cst.config, 'robust_model')
    else:
        robust_model = load_model(cst.ROBUST_trained_model)

    # Show the effect of learning only on attacked data
    print("On not attacked data, on the test data the robust model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test, y_test, robust_model)
    print("On attacked data, on the test data the robust model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test_attacked, y_test, robust_model)

    # Train or Load a large robust model to attacked data but also to not attacked data
    if cst.TRAIN_large_robust_model:
        print("Training a new only attack robust model")
        new_sample_train = np.concatenate((x_train_attacked, x_train), axis=0)
        new_label_train = np.concatenate((y_train, y_train), axis=0)
        large_robust_model = models.train_model(new_sample_train, new_label_train, cst.config, 'large_robust_model')
    else:
        large_robust_model = load_model(cst.LARGE_ROBUST_trained_model)

    # Show the effect of learning only on attacked data
    print("On not attacked data, on the test data the robust model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test, y_test, large_robust_model)
    print("On attacked data, on the test data the robust model has an accuracy of:")
    visualization.show_dataset_and_predictions(x_test_attacked, y_test, large_robust_model)
