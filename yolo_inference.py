from ultralytics import YOLO

model = YOLO('yolov8x')

prediction = model.track(r"C:\Users\salwa\OneDrive\Desktop\tennis project\input\input_video.mp4",conf=0.2 , save=True)
