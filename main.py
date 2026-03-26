import math
import pyrealsense2 as rs
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# open the RGBD file with pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file("tonifuentes.bag", repeat_playback=False)
config.enable_stream(rs.stream.color)
config.enable_stream(rs.stream.depth)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data()) / 1000  # units are in milimeters

        results = model.predict(source=color, imgsz=640, conf=0.25, iou=0.45, max_det=100)
        result = results[0]

        for box in result.boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0])
            cls = model.names[int(box.cls[0])]
            median_depth = np.median(depth[y0:y1, x0:x1]).item()
            max_vel = 5 * math.sqrt(9 + 4 * median_depth) - 15
            label = f"{cls} | {median_depth:.2f}m | {max_vel:.2f}km/h"

            # Draw box
            cv2.rectangle(color, (x0, y0), (x1, y1), (0, 255, 0), 2)

            # Draw label
            cv2.putText(
                color,
                label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

        # im_array_bgr = results[0].plot()  # BGR format
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)  # Convert to RGB

        cv2.imshow("Pretrained Model", color)

        # exit when the escape key is pressed
        if cv2.waitKey(1) == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
