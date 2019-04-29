import os

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
import tensorflow as tf
from PIL import Image

from model import cnn_model_fn

MODEL_PATH = os.getenv("MODEL_PATH")
HEIGHT, WIDTH = 299, 299

checkpointing_config = tf.estimator.RunConfig(
    save_checkpoints_steps=500,
    keep_checkpoint_max=200
)

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir=MODEL_PATH,
    config=checkpointing_config
)


def predict(file):
    img = convert_Image(file)
    x = predict_image(img)
    return {
        'classes': x["classes"],
        'predictions': list(x["probabilities"]),
        'max_prob': x["probabilities"].max()
    }


def convert_Image(file):
    img = Image.open(file)
    img = img.resize((HEIGHT, WIDTH), Image.ANTIALIAS)
    img = np.asarray(img) / np.float(255)
    img = np.asarray(img, dtype=np.float32).reshape(-1, HEIGHT, WIDTH, 3)
    return img


def predict_image(img):
    predict_input_fn2 = tf.estimator.inputs.numpy_input_fn(
        x={"x": img},
        num_epochs=1,
        shuffle=False)

    predict_results = mnist_classifier.predict(
        input_fn=predict_input_fn2)

    for x in predict_results:
        print(x)
        return(x)
