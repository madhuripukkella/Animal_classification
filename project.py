import cv2
import time

text=["",""]

# Pretrained classes in the model
classNames = {0: 'background',
              1: 'person', 16: 'bird', 17: 'cat',18: 'dog', 19: 'horse', 20: 'sheep',
              21: 'cow', 22: 'elephant', 23: 'bear',24: 'zebra', 25: 'giraffe'}


def id_class_name(class_id, classes):
    for key, value in classes.items():
        if class_id == key:
            return value


# Loading model
model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb',
                                      'ssd_mobilenet_v2_coco_2018_03_29.pbtxt')
while(1):
    #cap = cv2.VideoCapture(0)
    #check, frame = cap.read()
    #check, frame = cap.read()
    #check, frame = cap.read()
    frame = cv2.imread("test8.jpeg")
    frame = cv2.imread("test6.jpeg")
    frame = cv2.resize(frame, (640, 480))
#    print(check)
    image = frame
   
    image_height, image_width, _ = image.shape

    model.setInput(cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=True))
    output = model.forward()

    text[0]=""
    text[1]=""
    
    for detection in output[0, 0, :, :]:
        confidence = detection[2]
        if confidence > .5:
            class_id = detection[1]
            if class_id == 0 or class_id == 1 or class_id == 16 or class_id == 17 or class_id == 18 or class_id == 19 or class_id == 20 or class_id == 21 or class_id == 22 or class_id == 23 or class_id == 24 or class_id == 25 :
                class_name=id_class_name(class_id,classNames)
                if( (class_id != None) and (detection[2] != None) ):
                    print(str("OUTPUT : "+ str(class_id) + " " + str(detection[2])  + " " + class_name))
                box_x = detection[3] * image_width
                box_y = detection[4] * image_height
                box_width = detection[5] * image_width
                box_height = detection[6] * image_height
                cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                cv2.putText(image,class_name ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))
                if class_id == 1 :
                    text[0] = "People"
                if class_id > 15 and class_id < 26 :
                    text[1] = "Animal"

    text2 = ""

    if(text[0]!=""):
        text[0] = text[0].replace(' ', '_')
        text2 = text[0]+"_detected"
        
    if(text[1]!=""):
        text[1] = text[1].replace(' ', '_')
        text2 = text[1]+"_detected"

    if(text[0]!="" and text[1]!=""):
        text2 = text[0]+"_detected_and_"+ text[1]+"_detected"

    text[0]=""
    text[1]=""
    
    cv2.imwrite("image_box_text.jpg",image)
    imS = cv2.resize(image, (640, 480))
    cv2.imshow('Captured Image',imS)
    cv2.moveWindow('Captured Image',50,50)
    cv2.waitKey(1)
#    cap.release()
    
cv2.waitKey(0)
cv2.destroyAllWindows()
