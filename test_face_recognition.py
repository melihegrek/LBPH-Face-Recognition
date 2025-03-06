import cv2
import numpy as np
import os

# Yüz sınıflandırıcıyı yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Yüz tanıma modelini yükle
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")  # Eğitimden kaydedilen model

# Kişi isimlerini veri setinden al
dataset_path = 'lfw-deepfunneled'
person_names = os.listdir(dataset_path)  # LFW klasörlerinde isimleri alın

# Test fotoğrafını yükle
test_image = cv2.imread("testimg5.jpg")  # Test için kullanılan fotoğraf

# Fotoğrafı yeniden boyutlandır
test_image_resized = cv2.resize(test_image, (250, 250))  # 250x250 boyutuna

# Yüz algılama
gray = cv2.cvtColor(test_image_resized, cv2.COLOR_BGR2GRAY)  # Boyutlandırılmış fotoğrafın gri tonlamasına çevir
faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces_detected:
    # Yüz resmini kes
    face = gray[y:y+h, x:x+w]
    
    # Yüz tanıma
    label, confidence = recognizer.predict(face)  # predict ile modelin tahminini alıyoruz
    
    # Tanıma sonuçlarını yazdır
    print(f"Tanınan Kişi: {person_names[label]}")  # person_names dizisindeki isme göre kişiyi yazdır
    print(f"Güven Düzeyi: {confidence}")  # Modelin tahmininin güven seviyesini yazdır
    
    # Test resmini göster (yüzü dikdörtgenle işaretle)
    cv2.rectangle(test_image_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Yüzü dikdörtgenle işaretle
    cv2.putText(test_image_resized, f"{person_names[label]} - {round(confidence, 2)}", 
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Farklı pencerelerde gösterim
cv2.imshow("Grayscale Image", gray)  # GrayScale hali
cv2.imshow("Face Detected", test_image_resized)  # Yüz algılamalı hali

# Test resmini yüz tanımlamasıyla birlikte göster
cv2.waitKey(0)
cv2.destroyAllWindows()
