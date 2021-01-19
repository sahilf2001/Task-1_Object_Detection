import cv2,glob

all_images = glob.glob("*.jpg")


classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

for image in all_images:
    img = cv2.imread(image)
    classIds, confs, bbox = net.detect(img,confThreshold=0.5)

    #print(classIds,bbox)

    for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
        cv2.rectangle(img,box,color=(0,0,255),thickness=3)
        cv2.putText(img,classNames[classId - 1].upper(),(box[0] + 10,box[1] + 30),
                cv2.FONT_HERSHEY_DUPLEX,1,(255, 0, 0),2)



    cv2.imshow("Output",img)
    cv2.waitKey(3000)
