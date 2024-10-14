import cv2
import mediapipe as mp
import math

# Mediapipe ishlatish uchun asosiy ob'ektlar
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Kamera va odam orasidagi masofa (2 metr)
known_distance = 4.0  # metr

# Taxminiy kameraning fokus uzunligi (bu kalibrlash orqali aniqlanadi, taxminan)
focal_length = 1100  # taxminiy pikseldagi qiymat

# Video yoki tasvirni yuklash
cap = cv2.VideoCapture('http://192.168.233.74:8080/video')  # Kameradan video olish yoki video fayl yuklash mumkin

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Rasmni BGR dan RGB ga o‘tkazish
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Pose Estimation bajarish
    results = pose.process(image)

    # Tasvirni qayta yozish (pose ni chizish uchun)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Agar odam topilsa
    if results.pose_landmarks:
        # Pose nuqtalarini chizish
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Bosh (nozdralar) va oyoq nuqtalarini olish
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        left_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL]
        right_heel = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL]

        # O‘rtacha oyoq pozitsiyasini hisoblash
        heel_y = (left_heel.y + right_heel.y) / 2

        # Odam bo‘yi (ekran koordinatalarida, normalized)
        height = abs(nose.y - heel_y)

        # Ekran balandligi bo'yicha o'lchash (pikselda)
        image_height, image_width, _ = image.shape
        pixel_height = height * image_height  # Normalize qilingan y-ni piksellarga aylantirish

        # Odamning haqiqiy bo'yini hisoblash (yuqoridagi formula orqali)
        real_height = (pixel_height * known_distance) / focal_length

        # Bo‘yni ekranga chiqarish
        cv2.putText(image, f'Height (Real): {real_height:.2f} meters', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Tasvirni ko‘rsatish
    cv2.imshow('MediaPipe Pose', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()