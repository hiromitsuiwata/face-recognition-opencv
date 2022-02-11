import cv2

cascade_path = "haarcascade_frontalface_default.xml"

input_path = "./images/input.jpg"
output_path = "./images/output.jpg"

image = cv2.imread(input_path)

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cascade = cv2.CascadeClassifier(cascade_path)

facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

white = (255, 255, 255)

if len(facerect) > 0:
    for rect in facerect:
        cv2.rectangle(image, tuple(rect[0:2]), tuple(rect[0:2] + rect[2:4]), white, thickness=2)
    cv2.imwrite(output_path, image)