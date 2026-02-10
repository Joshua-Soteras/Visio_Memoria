import cv2
from ultralytics import YOLO
import os 


#Dynamic Pathing 
#gets the directory where this script exist 
current_dir = os.path.dirname(os.path.abspath(__file__))

#construct the path dyanamically 
model_path = os.path.join(current_dir, "yolov8-face", "yolov8n-face.pt")

model = YOLO(model_path)


# --- DEBUGGING SECTION ---
print("Attempting to open camera...")
cap = cv2.VideoCapture(0) # Try 0, then 1 if this fails

if not cap.isOpened():
    print("ERROR: Could not open video source.")
    exit()
else:
    print("Camera initialized successfully.")

print("Starting loop. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()
    
    cv2.imshow("Visio_Memoroia", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()