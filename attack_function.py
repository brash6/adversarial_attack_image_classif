import constants as cst
import visualization

import numpy as np
from tqdm import tqdm_notebook
import os
import tensorflow as tf
import random
from art.classifiers import KerasClassifier
from art.attacks import FastGradientMethod, CarliniLInfMethod, DeepFool, CarliniL2Method


def FGSM(x_test, y_test, delta, model):
    index = np.argmax(y_test)
    x = tf.constant(np.expand_dims(x_test, 0), dtype=tf.float32)
    y = tf.one_hot(index, 10)
    y = tf.reshape(y, (1, 10))

    with tf.GradientTape() as tape:
        tape.watch(x)
        prediction = model(x)
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        loss = loss_object(y, prediction)
        gradient = tape.gradient(loss, x)
        signed_grad = tf.sign(gradient)
        x_attack = x + delta * np.squeeze(signed_grad, 0)
        x_attack = np.clip(x_attack, 0, 1)  # car on a des valeurs comprises entre 0 et 1 et non 255
    return np.reshape(x_attack, (32, 32, 3))


def PGD_infini(x_test, y_test, delta, epsilon, num_iter, model):
    index = np.argmax(y_test)
    x = tf.constant(np.expand_dims(x_test, 0), dtype=tf.float32)
    y = tf.one_hot(index, 10)
    y = tf.reshape(y, (1, 10))
    x_attack = x
    with tf.GradientTape() as g:
        g.watch(x)
        prediction = model(x)
        loss_func = tf.keras.losses.CategoricalCrossentropy()
        loss_value = loss_func(y, prediction)
        gradient = g.gradient(loss_value, x)
        signed_grad = np.sign(gradient)
        for i in range(num_iter):
            x_attack = x_attack + delta * signed_grad
            eta = tf.clip_by_value(x_attack - x, -epsilon, epsilon)
            x_attack = tf.clip_by_value(x_attack + eta, 0, 1)

    return np.reshape(x_attack, (32, 32, 3))


def make_attack(x, y, delta, epsilon, model, name, num_iter=3, style='PGD'):
    if style == 'PGD':
        attacked_data = np.array(
            [PGD_infini(x[idx], y[idx], delta, epsilon, num_iter, model) for idx in tqdm_notebook(range(len(x)))])
        np.save(os.path.join(cst.DATA, name), attacked_data)
    if style == 'FGSM':
        attacked_data = np.array(
            [FGSM(x[idx], y[idx], delta, model) for idx in tqdm_notebook(range(len(x)))])
        np.save(os.path.join(cst.DATA, name), attacked_data)
    return attacked_data


def attack_gen(x_train, y_train, model, batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            index = random.randint(0, 49999)
            if cst.attack_style == "PGD":
                x.append(
                    PGD_infini(x_train[index], y_train[index], cst.attack_delta, cst.attack_epsilon, cst.attack_nb_iter,
                               model))
                y.append(y_train[index])
            if cst.attack_style == 'FGSM':
                x.append(FGSM(x_train[index], y_train[index], cst.attack_delta, model))
                y.append(y_train[index])
        x = np.array(x)
        y = np.array(y)

        yield (x, y)


def attack_gen_rand(x_train, y_train, model, batch_size, attack_delta=0.003, attack_nb_iter=3, attack_style="FGSM"):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            index = random.randint(0, 49999)
            tresh = random.randint(0, 100)
            if tresh > 50:
                if attack_style == "PGD":
                    x.append(PGD_infini(x_train[index], y_train[index], attack_delta, attack_nb_iter, model))
                    y.append(y_train[index])
                if attack_style == 'FGSM':
                    x.append(FGSM(x_train[index], y_train[index], attack_delta, model))
                    y.append(y_train[index])
            else:
                x.append(x_train[index])
                y.append(y_train[index])
        x = np.array(x)
        y = np.array(y)

        yield (x, y)


def carlini_inf(x_test, model):
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    attack_cw = CarliniLInfMethod(classifier=classifier, eps=0.03, max_iter=40, learning_rate=0.01)
    x_test_adv = attack_cw.generate(x_test)
    return np.reshape(x_test_adv, (32, 32, 3))


def carlini_l2(x_test, model):
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    attack_cw = CarliniL2Method(classifier=classifier, confidence=0.0, targeted=False, learning_rate=0.01,
                                binary_search_steps=10, max_iter=10, initial_const=0.01, max_halving=5, max_doubling=5,
                                batch_size=1)
    x_test_adv = attack_cw.generate(x_test)
    return np.reshape(x_test_adv, (32, 32, 3))


def deep_fool(x_test, model):
    classifier = KerasClassifier(model=model, clip_values=(0, 1))
    attack_cw = DeepFool(classifier=classifier, max_iter=4, epsilon=0.03, nb_grads=10, batch_size=1)
    x_test_adv = attack_cw.generate(x_test)
    return np.reshape(x_test_adv, (32, 32, 3))
