import cv2
import numpy as np
from keras.models import load_model


class FaceClassifier:
    def __init__(self, model_path):
        self.model = load_model(model_path)
        self.classes = ['shon', 'no_shon']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def classify_face(self, image_path):
        image_path = image_path.strip('"')

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            resized_face = cv2.resize(face_img, (64, 64))
            resized_face = np.array(resized_face).reshape(-1, 64, 64, 1)

            prediction = self.model.predict(resized_face)
            predicted_class = self.classes[np.argmax(prediction)]
            if predicted_class == 'shon':
                return 'shon'

        return 'no_shon'


if __name__ == "__main__":
    model_path = '../model/modelo.h5'
    classifier = FaceClassifier(model_path)

    image_path = input("Ingrese la ruta de la imagen: ")
    predicted_class = classifier.classify_face(image_path)

    print(f'La clase predicha es: {predicted_class}')
