from cv2 import *

namedWindow("Camara")
vc=VideoCapture(1)

while True:
    next, frame=vc.read()
    imshow("Camara",frame)
    if waitKey(50)>=0:
        break