from __future__ import absolute_import, division, print_function

import os

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.INFO)

# Model Vars
MODEL_PATH = os.getenv("MODEL_PATH")
DATA_DIR = 'C:\\Users\\PC\\Downloads\\test_Conv\\Convolutional Models\\DatasetCreation\\data_shoes\\new_set\\'
HUB_INCEPTION_V3 = "https://tfhub.dev/google/imagenet/inception_v3/classification/1"

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 50
GLOBAL_STEPS = 1000
LEARNING_RATE = 1e-4


def cnn_model_fn(features, labels, mode):
    # Load Inception-v3 model.
    module = hub.Module(HUB_INCEPTION_V3, trainable=True)

    input_layer = tf.reshape(features["x"], [-1, 299, 299, 3])

    # Load Inception-v3 with layers signatures
    outputs = module(dict(images=input_layer),
                     signature="image_classification",
                     as_dict=True)

    # Bottleneck layer to start FC
    # INPUTS
    middle_output = outputs["InceptionV3/Mixed_7c"]

    # Global Average Pooling
    # Input  8x8x2048
    # Output 1x2048
    avgPool = tf.reduce_mean(middle_output, axis=[1, 2])

    # Batch Normalization
    bn1 = tf.layers.batch_normalization(
        avgPool, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense, FC
    # Input  1x2048
    # Output 1x128
    logits1 = tf.layers.dense(inputs=avgPool, units=128,
                              activation=tf.nn.relu)
    # Dropout
    dropout1 = tf.layers.dropout(
        inputs=logits1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense, FC
    # Input  1x128
    # Output 1x64
    logits2 = tf.layers.dense(inputs=dropout1, units=64,
                              activation=tf.nn.relu)
    # Dropout 2
    dropout2 = tf.layers.dropout(
        inputs=logits2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Dense, FC, OUTPUT
    # Input  1x128
    # Output 1x64
    logits = tf.layers.dense(inputs=dropout2, units=3)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        summary_hook = tf.train.SummarySaverHook(
            100,
            output_dir=MODEL_PATH + '\\tf',
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op,
                                          training_hooks=[summary_hook])

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    with tf.Graph().as_default() as g:
            # Load training and eval data
        ftrain = h5py.File(DATA_DIR + 'train_dataset_1_.h5', 'r')
        ftest = h5py.File(DATA_DIR + 'test_dataset_1_.h5', 'r')

        train_data, train_labels = ftrain['train_set_x'],  ftrain['train_set_y']
        eval_data, eval_labels = ftest['test_set_x'], ftest['test_set_y']

        train_data = np.asarray(train_data, dtype=np.float32)
        train_data = train_data/255.
        train_labels = np.asarray(
            train_labels, dtype=np.int32).reshape(train_labels.shape[0])

        eval_data = np.asarray(eval_data, dtype=np.float32)
        eval_data = eval_data/255.
        eval_labels = np.asarray(
            eval_labels, dtype=np.int32).reshape(eval_labels.shape[0])

        # Testing purposes...
        # plt.imshow(train_data[21])
        # plt.show()

        # Create the Estimator
        mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=MODEL_PATH)

        # Train the model
        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=BATCH_SIZE,
            num_epochs=EPOCHS,
            shuffle=True)

        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        for _ in range(EPOCHS):
            mnist_classifier.train(
                input_fn=train_input_fn,
                steps=GLOBAL_STEPS)

            eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
            print(eval_results)


if __name__ == "__main__":
    tf.app.run()
