from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.initializers import HeNormal
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.optimizers.schedules import ExponentialDecay
from keras.layers import Dropout
from sklearn.base import BaseEstimator, ClassifierMixin
from keras.losses import BinaryCrossentropy 
from sklearn.metrics import recall_score
import numpy as np


class B3D3AD_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = np.array([0, 1])  
        self.model = Sequential([
            self.DenseLayer(4096, activation='relu'),
            self.DenseLayer(512, activation='relu'),
            self.DenseLayer(512, activation='relu'),
            self.DenseLayer(512, activation='relu'),
            self.DropoutLayer(0.3),
            self.DenseLayer(1, activation='sigmoid'),
        ])

    # Customer Dense layer
    def DenseLayer(self, nodes, activation='relu'):
        return Dense(
            nodes, activation=activation, 
            kernel_initializer=HeNormal(), bias_initializer=HeNormal(),
            kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
        )

    # Custom dropout layer
    def DropoutLayer(self, rate):
        return Dropout(rate)

    # Resets weights to HeNormal
    def reset_weights(self):
        initial_weights = self.model.get_weights()
        self.model.set_weights(initial_weights)

    def predict(self, X, threshold=0.5):
        # Predict probabilities
        probabilities = self.model.predict(X)
        # Convert probabilities to binary predictions using the threshold
        predictions = (probabilities >= threshold).astype(int)
        return predictions

    # compile the model
    def compile(self):
        self.model.compile(optimizer='adam', loss=BinaryCrossentropy(), metrics=['accuracy'])

    # Calculate recall
    def recall(self, X, y):
        predictions = self.predict(X)
        return recall_score(y, predictions)

    # Run the model. Forward fit using a learning rate scheduler
    def fit(self, X, training_labels, epochs=5, batch_size=32):
        lr_scheduler = ExponentialDecay(initial_learning_rate=0.001, decay_steps=1, decay_rate=.1)
        self.compile()
        self.model.fit(X, training_labels, epochs=epochs, 
                    batch_size=batch_size, callbacks=[LearningRateScheduler(lr_scheduler)])
