# Basic usage
from ultralytics import YOLO

#load the model 
#
# this is a pose estimation model
model = YOLO("yolov8n-face.pt")

#Load results 
#model() denotes callable object 
#preprocesss input and returns Results() object(s)
#there can be multiple Results objects 
results = model("image.jpg") # on an image 
results = model(source = 0, show = True, confe = 0.25)


'''
the landmarks are what you then use to align the face before DINOv3:
left_eye, right_eye, nose, left_mouth, right_mouth → affine transform → 112x112 aligned crop → DINOv3 embeddingfor r in results:
'''

'''
    .xyxy coordinates of the boxes in cv 
    .conf the confidence score 
    .cls classID 
    .keypoints -> landmarks
        - keypoints are eyes left/right,  nose and motuh 
        - .keypoints.xy is the coordinates of special landmark

    results.boxes.xyxy: 
        - A Tensor containing the bounding box coordinates for all detected faces. 
        - If 3 faces were found, this has 3 rows.
        - results.keypoints.xy: A Tensor containing the landmark coordinates (eyes, nose, etc.) for all detected faces.
          If 3 faces were found, this also has 3 rows.
        - pixel coordinates must be whole numbers
'''


for box, kpts in zip(results.boxes.xyxy, results.keypoints.xy):

    #.int -> convert pixel coordinates to whole numbers, coordinates must be whole numebrs
    #.tolist coverting pytorrch tensor in python list
    #get the four values left top right and bottom 
    x1, y1, x2, y2 = box.int().tolist()

    #kpts represents a specific face
    #Tensor with shape of (5,2 )
    landmarks = kpts.tolist()  # 5 points: left_eye, right_eye, nose, mouth_l, mouth_r
    print(f"Face at ({x1},{y1})-({x2},{y2})")
    print(f"Landmarks: {landmarks}")