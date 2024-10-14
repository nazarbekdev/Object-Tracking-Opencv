import cv2
import numpy as np

# Kameradan video olish
cap = cv2.VideoCapture(0)

# To'rtburchak maydoni uchun rangni belgilash (HSV formatida)
lower_color = np.array([30, 150, 50])  # Pastki rang (yashil rang misoli)
upper_color = np.array([85, 255, 255])  # Yuqori rang (yashil rang misoli)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Tasvirni BGR dan HSV ga o‘tkazish
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rang diapazonini filtr orqali to'rtburchakni aniqlash
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Konturlarni aniqlash
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Har bir kontur uchun
    for contour in contours:
        # Agar konturning maydoni katta bo'lsa
        if cv2.contourArea(contour) > 50:
            # To'rtburchakning o'lchovlarini olish
            x, y, w, h = cv2.boundingRect(contour)

            # Obyekt atrofida kvadrat chizish
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Obyektning koordinatalarini ekranga chiqarish
            cv2.putText(frame, f'Coords: ({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tasvirni ko‘rsatish
    cv2.imshow('Detected Rectangle', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
