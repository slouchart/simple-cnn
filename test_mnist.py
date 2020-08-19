
import mnist
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.utils import to_categorical


TRAIN_IMAGES = 0
TRAIN_LABELS = 1
TEST_IMAGES = 2
TEST_LABELS = 3


def prep_sets():
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()
    test_images = mnist.test_images()
    test_labels = mnist.test_labels()
    return train_images, train_labels, test_images, test_labels


def normalize(data):
    return (data[TRAIN_IMAGES] / 255) - 0.5, data[TRAIN_LABELS], \
           (data[TEST_IMAGES] / 255) - 0.5, data[TEST_LABELS]


def reshape(data):
    return np.expand_dims(data[TRAIN_IMAGES], axis=3), data[TRAIN_LABELS], \
           np.expand_dims(data[TEST_IMAGES], axis=3), data[TEST_LABELS]


def create_model(input_shape, num_filters=8, filter_size=3, pool_size=2):
    return Sequential(
        [
            Conv2D(num_filters, filter_size, input_shape=input_shape),
            MaxPooling2D(pool_size=pool_size),
            Flatten(),
            Dense(10, activation='softmax')
        ]
    )


def train_model(model, data):

    model.compile('adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(
        data[TRAIN_IMAGES],
        to_categorical(data[TRAIN_LABELS]),
        epochs=3,
        validation_data=(data[TEST_IMAGES], to_categorical(data[TEST_LABELS]))
    )
    return model


if __name__ == '__main__':
    data = prep_sets()
    data = normalize(data)
    data = reshape(data)
    should_train = False
    model = create_model(data[TRAIN_IMAGES].shape[1:])
    if should_train:
        model = train_model(model, data)
        model.save_weights('cnn.h5')

    model.load_weights('cnn.h5')
    model.summary()
    predictions = model.predict(data[TEST_IMAGES][:5])

    print(np.argmax(predictions, axis=1))
    print(data[TEST_LABELS][:5])


