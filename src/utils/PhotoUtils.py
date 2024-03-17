import os
import shutil
import time

import cv2


class PhotoUtil:
    def __init__(self, cascade_path, output_folder):
        self.output_folder = output_folder
        self.window_name = 'Window'
        cv2.namedWindow(self.window_name)
        self.cap = cv2.VideoCapture(0)
        self.detector = cv2.CascadeClassifier(cascade_path)
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        os.makedirs(output_folder)

    def detect_and_show_face(self):
        i = 0
        while True:
            ret, image = self.cap.read()
            mirrored_image = cv2.flip(image, 1)
            faces = self.detector.detectMultiScale(mirrored_image, 1.3, 5)

            for (x, y, w, h) in faces:
                # Recortar la región del rostro
                face_cropped = mirrored_image[y:y + h, x:x + w]
                # Mostrar la región del rostro recortado en la ventana
                cv2.imshow(self.window_name, face_cropped)
                if i <= 100:
                    cv2.imwrite(os.path.join(self.output_folder, f'face_{int(time.time())}_{i}.jpg'), face_cropped)
                    i += 1
                else:
                    print("Se tomaron correctamente las 100 fotos")

            if cv2.waitKey(1) & 0xFF == 27:
                break

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    util = PhotoUtil(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml', '../data/shonFace')
    util.detect_and_show_face()
    util.release()
