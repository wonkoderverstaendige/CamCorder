import csv
import cv2
import sys
from pathlib import Path

path = Path(sys.argv[1]).resolve()
cap = cv2.VideoCapture(str(path))

def mp_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y, flags, param)


cv2.namedWindow('video')
cv2.setMouseCallback('video', mp_callback)

while True:
    rt, frame = cap.read()
    if not rt:
        break

    cv2.imshow('video', frame)
    key = cv2.waitKey(66)
    if key == ord('q'):
        break
