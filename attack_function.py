import constants as cst

import numpy as np
from tqdm import tqdm_notebook
import os
import tensorflow as tf


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
        x_attack = x_test + delta * np.squeeze(signed_grad, 0)
        x_attack = np.clip(x_attack, 0, 1)  # car on a des valeurs comprises entre 0 et 1 et non 255
    return x_attack


def PGD_infini(x_test, y_test, delta, num_iter, model):
    index = np.argmax(y_test)
    x = tf.constant(np.expand_dims(x_test, 0), dtype=tf.float32)
    y = tf.one_hot(index, 10)
    y = tf.reshape(y, (1, 10))
    x_attack = x
    for i in range(num_iter):
        with tf.GradientTape() as g:
            g.watch(x)
            prediction = model(x)
            loss_func = tf.keras.losses.CategoricalCrossentropy()
            loss_value = loss_func(y, prediction)
            gradient = g.gradient(loss_value, x)
            signed_grad = np.sign(gradient)
            x_attack = x_attack + delta * signed_grad
            eta = tf.clip_by_value(x_attack - x, -delta, delta)
            x_attack = tf.clip_by_value(x + eta, 0, 1)

    return np.reshape(x_attack, (32, 32, 3))


def make_attack(x, y, delta, num_iter, model, name, style='PGD'):
    if style == 'PGD':
        attacked_data = np.array([PGD_infini(x[idx], y[idx], delta, num_iter, model) for idx in tqdm_notebook(range(len(x)))])
        np.save(os.path.join(cst.DATA, name), attacked_data)
    if style == 'FGSM':
        attacked_data = np.array(
            [FGSM(x[idx], y[idx], delta, model) for idx in tqdm_notebook(range(len(x)))])
        np.save(os.path.join(cst.DATA, name), attacked_data)
    return attacked_data

