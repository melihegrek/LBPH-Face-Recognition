import cv2
import os
import numpy as np

# Yüz sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Eğitim için veri (yüzler ve etiketler)
faces = []
labels = []

# LFW veri setindeki kişilerin isimlerini belirleyin
dataset_path = 'lfw-deepfunneled'  # LFW veri setinin dizini
person_names = os.listdir(dataset_path)  # LFW klasörlerinde isimleri alın

# Eğitim verilerini toplama
for label, person in enumerate(person_names):
    person_path = os.path.join(dataset_path, person)
    
    # Kişinin fotoğraflarını oku
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        
        # Fotoğrafı oku
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Yüz algılama
        faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces_detected:
            # Yüz resmini kes
            face = gray[y:y+h, x:x+w]
            faces.append(face)
            labels.append(label)

# LBPH ile yüz tanıma modelini eğit
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Modeli kaydet
recognizer.save("face_recognizer.yml")
print("Yüz tanıma eğitimi tamamlandı ve model kaydedildi!")
