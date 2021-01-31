import os

import albumentations as A

import numpy as np
import pandas as pd

from skimage.io import imread

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
import tensorflow.keras.layers as L
from tensorflow.keras.metrics import MeanSquaredError
from tensorflow.keras.optimizers import Adam


AUTOTUNE = tf.data.experimental.AUTOTUNE


def get_model():
    model = Sequential()

    model.add(
        L.Conv2D(
            3,
            32,
            padding='same',
            input_shape=(100, 100, 3)
        )
    )
    for filters in [None, 64, 128, 256, 512]:
        if filters:
            model.add(L.Conv2D(filters, 3, padding='same'))
            
        model.add(L.BatchNormalization())
        model.add(L.ReLU())
        model.add(L.MaxPool2D())

    model.add(L.Flatten())
    model.add(L.Dropout(0.3))
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(1000))
    model.add(L.BatchNormalization())
    model.add(L.ReLU())
    model.add(L.Dense(28))
    model.compile(
        optimizer=Adam(),
        loss='mse',
    )
    return model

def get_transforms():
    train_transform = A.Compose(
        [
            A.Resize(width=100, height=100, always_apply=True),
            A.Rotate(limit=30, p=0.5),
            A.Normalize(always_apply=True)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )

    val_transform = A.Compose(
        [
            A.Resize(width=100, height=100, always_apply=True),
            A.Normalize(always_apply=True)
        ],
        keypoint_params=A.KeypointParams(format='xy', remove_invisible=True)
    )
    return train_transform, val_transform

def parse_sample(
    sample,
    dir_path
):
    num_str = sample['filename']
    coordinates = sample.values[1:]
    points = np.array(
        [
            (coordinates[2 * i], coordinates[2 * i + 1])
            for i in range(14)
        ]
    )
    return dir_path + num_str, points


def dataset_gen(
    ds,
    dir_path
):
    def gen():
        for i in range(ds.shape[0]):
            sample = ds.iloc[i]
            img_path, points = parse_sample(sample, dir_path=dir_path)
            yield img_path, points
    return gen


def test_dataset_gen(
    ds,
    dir_path
):
    def gen():
        for filename in os.listdir(dir_path):
            img_path = os.path.join(dir_path, filename)
            points = np.ones([14, 2]) * 20
            yield img_path, points
    return gen


@tf.autograph.experimental.do_not_convert
def load():
    def f(
        img_path,
        kp
    ):
        img = imread(img_path.decode('utf-8'))
        if len(img.shape) == 2:
            img = np.dstack([img, ]*3)
        return img.astype(np.float32), kp.astype(np.float32)
    return f


@tf.autograph.experimental.do_not_convert
def preprocess(
    transform,
    reserve_transform,
    num_trials
):
    def f(
        img,
        kp
    ):
        kp = np.clip(kp, 0, None)
        for t in range(num_trials):
            transformed = transform(image=img, keypoints=kp)
            if len(transformed['keypoints']) == 14:
                break
        else:
            transformed = reserve_transform(image=img, keypoints=kp)
        return transformed['image'].astype(
            np.float32
        ), np.array(
            transformed['keypoints']
        ).reshape(-1).astype(np.float32)

    return f

def loader(x, y):
    return tf.numpy_function(
        func=load(),
        inp=[x, y],
        Tout=(tf.float32, tf.float32)
    )

def create_dataset(
    ds,
    dir_path,
    transform,
    reserve_transform,
    dataset_gen=dataset_gen,
    num_trials=20,
    batch_size=5,
    shuffle_buffer_size=200,
    prefetch_size=AUTOTUNE,
    num_calls=AUTOTUNE,
    repeat=True,
    shuffle=True,
):
    gen = dataset_gen(ds, dir_path=dir_path)
    dataset = tf.data.Dataset.from_generator(
        gen,
        (tf.string, tf.float32),
        output_shapes=(tf.TensorShape([]), tf.TensorShape([14, 2]))
    )

    dataset = dataset.map(
        loader,
        num_parallel_calls=num_calls
    )

    def preprocessor(x, y):
        return tf.numpy_function(
            func=preprocess(transform,
                            reserve_transform,
                            num_trials=num_trials),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        )
    dataset = dataset.map(
        preprocessor,
        num_parallel_calls=num_calls
    )

    def set_shapes(img, kp, img_shape=(100, 100, 3)):
        img.set_shape(img_shape)
        kp.set_shape([28])
        return img, kp
    dataset = dataset.map(set_shapes, num_parallel_calls=num_calls)
    if repeat:
        dataset = dataset.repeat()
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset



def train_detector(train_gt, train_img_dir, fast_train=False):
    if fast_train:
        return
    train_gt = train_gt.iloc[:20]
    train_transform, val_transform = get_transforms()
    train_dataset = create_dataset(
        train_gt,
        train_img_dir,
        train_transform,
        val_transform,
        num_trials=1,
        batch_size=1,
        shuffle=False,
        repeat=False,
        prefetch_size=1,
        num_calls=1
    )
    steps_per_epoch = np.ceil(train_gt.shape[0] / 1).astype(int)

    model = get_model()

    model.fit(
        train_dataset,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
    )
    return model


def read_img_shapes(filenames, img_dir):
    img_shapes = []
    for fname in filenames:
        img_shapes.append(imread(os.path.join(img_dir, fname)).shape[:2])
    return np.asarray(img_shapes)


def detect(model, test_img_dir):
    filenames = os.listdir(test_img_dir)
    test_bs = 4
    test_dataset = create_dataset(
        None,
        test_img_dir,
        val_transform,
        val_transform,
        dataset_gen=test_dataset_gen,
        num_trials=1,
        batch_size=test_bs,
        prefetch_size=1,
        num_calls=1,
        repeat=False,
        shuffle=False
    )
    pred = model.predict(
        test_dataset,
        steps=np.ceil(len(filenames) / test_bs)
    )
    shapes = read_img_shapes(filenames, test_img_dir)
    pred[:, ::2] *= shapes[:, 1:] / 100
    pred[:, 1::2] *= shapes[:, :1] / 100
    return {f: p for f, p in zip(filenames, pred)}
