import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class ImageClassifier:
    def __init__(self, data_directory, classes, model_directory, char_directory, img_rows=64, img_cols=64, test_size=0.2):
        self.data_directory = data_directory
        self.classes = classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.test_size = test_size
        self.model_directory = model_directory
        self.char_directory = char_directory
        self.epochs=30

    def load_data(self):
        data = []
        target = []

        for clase in self.classes:
            folder_path = os.path.join(self.data_directory, clase)
            for img in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (self.img_rows, self.img_cols))
                data.append(np.array(image))
                target.append(self.classes.index(clase))
        data = np.array(data)
        data = data.reshape(data.shape[0], self.img_rows, self.img_cols, 1)
        target = np.array(target)
        new_target = to_categorical(target, len(self.classes))
        return data, new_target

    def train_model(self):
        data, target = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=self.test_size)

        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.classes), activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=40, epochs=self.epochs, verbose=1, validation_data=(X_test, y_test))

        model.save(f'{model_directory}/modelo.h5')

        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)
        confusion_mtx = confusion_matrix(y_true, y_pred_classes)

        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mtx, annot=True, fmt='g', cmap='Blues', xticklabels=self.classes,
                    yticklabels=self.classes)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.savefig(f'{self.char_directory}/matriz_confusion.png')
        plt.show()

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Historial de Error')
        plt.ylabel('Error')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
        plt.savefig('graficas/historial_error.png')
        plt.show()


if __name__ == "__main__":
    data_directory = '../data/entrenamiento'
    char_directory = '../data/graficas'
    model_directory = '../data/model'
    classes = ['shon', 'no_shon']
    classifier = ImageClassifier(data_directory, classes, model_directory, char_directory)
    classifier.train_model()
