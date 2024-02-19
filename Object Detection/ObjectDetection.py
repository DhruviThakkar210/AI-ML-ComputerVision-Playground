import cv2
import pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("veh1.mp4")

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
vehicle_classes = ["car", "truck", "bus", "motorcycle"]

while True:
    count += 1
    if count % 4 != 0:
        continue
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        if c in vehicle_classes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()