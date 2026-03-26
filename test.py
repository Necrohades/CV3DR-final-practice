from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import pyrealsense2 as rs

model = YOLO("yolo11n.pt")

fig, axes = plt.subplots(1, 1, figsize=(12, 10))
fig.suptitle("Pretrained Model", fontsize=16)


results = model.predict(source="mug.jpeg", imgsz=640, conf=0.25, iou=0.45, max_det=100)
print(results)
im_array_bgr = results[0].plot()  # BGR format
im_array_rgb = cv2.cvtColor(im_array_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB
plt.imshow(im_array_rgb)
plt.show()
