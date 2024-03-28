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
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import applications

def get_file_size(file_path):
    size = os.path.getsize(file_path)
    return size

def convert_bytes(size, unit=None):
    if unit == "KB":
        return print('File size: ' + str(round(size / 1024, 3)) + ' Kilobytes')
    elif unit == "MB":
        return print('File size: ' + str(round(size / (1024 * 1024), 3)) + ' Megabytes')
    else:
        return print('File size: ' + str(size) + ' bytes')

class ImageClassifier:
    def __init__(self, data_directory, classes, model_directory, char_directory, name, img_rows=64, img_cols=64,
                 test_size=0.2):
        self.data_directory = data_directory
        self.classes = classes
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.test_size = test_size
        self.model_directory = model_directory
        self.char_directory = char_directory
        self.name = name
        self.epochs = 50

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
        model.add(Conv2D(32, kernel_size=(3, 3), activation='tanh', input_shape=(self.img_rows, self.img_cols, 1)))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.classes), activation='softmax'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=40, epochs=self.epochs, verbose=1,
                            validation_data=(X_test, y_test))

        export_path = f'{self.model_directory}{self.name}'
        model.save(export_path)
        self.convert_to_tflite(export_path)

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

    def convert_to_tflite(self, export_path):

        # Cargar el modelo guardado
        model = load_model(export_path, compile=False)

        TF_LITE_MODEL_FILE_NAME = f"{self.model_directory}/tflite_model.tflite"
        tf_lite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = tf_lite_converter.convert()
        tflite_model_name = TF_LITE_MODEL_FILE_NAME
        open(tflite_model_name, "wb").write(tflite_model)
        convert_bytes(get_file_size(TF_LITE_MODEL_FILE_NAME), "KB")

        # Save the model.
        with open(f'{self.model_directory}/tflite_model_another.tflite', 'wb') as f:
            f.write(tflite_model)


if __name__ == "__main__":
    name = "/modelo.h5"
    path = "../data/model"
    path_complete = path + name
    data_directory = '../data/entrenamiento'
    char_directory = '../data/graficas'
    model_directory = '../data/model'
    classes = ['shon', 'no_shon']
    classifier = ImageClassifier(data_directory, classes, model_directory, char_directory, name)
    classifier.train_model()
    classifier.convert_to_tflite(path_complete)