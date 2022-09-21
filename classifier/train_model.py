import json
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
import numpy as np


DATASET_PATH = "speech_data.json"
SAVE_MODEL_PATH = "model.h5"

LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 16
NUM_KEYWORDS = 12


def load_data(path):
    
    with open(path, "r") as fp:
        data_in = json.load(fp)
        
    X = np.array(data_in["mfcc"])
    y = np.array(data_in["labels"])
    
    return X, y


def prepare_datasets(path, validation_size, test_size):
    X, y = load_data(path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)
    
    # cnn tensorflow expects a 3d array
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, y_train, y_validation, y_test


def build_model(input_shape, lr, error="sparse_categorical_crossentropy"):
    
    model = keras.Sequential()
    
    # 1st conv layer
    model.add(keras.layers.Conv2D(128, (3, 3),
                                  activation='relu',
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(keras.layers.Conv2D(64, (3, 3),
                                  activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(keras.layers.Conv2D(32, (2, 2),
                                  activation='relu',
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))
    
    # flatten -> dense layer 1
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64,
                                 activation="relu",
                                 kernel_regularizer=keras.regularizers.L2(0.001)))
    model.add(keras.layers.Dropout(0.3))
    
    
    # sofmax output classifier
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))
    
    # compile modek
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=error, metrics=["accuracy"])
    
    # pring summary
    model.summary()
    
    
    return model



if __name__ == "__main__":
    # split up data
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(DATASET_PATH, 0.1, 0.2)
    
    # compile model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # coefficients, 1)
    model = build_model(input_shape, LEARNING_RATE)
    
    # fit model
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_data=(X_validation, y_validation))
    
    # evaluate model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Error: {test_error}, Test Accuracy: {test_accuracy}")
    
    # save model
    model.save(SAVE_MODEL_PATH)
