import cv2


def centroid(cnt):
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return cx, cy


class Tracker:
    def __init__(self, idx=0):
        super().__init__()
        self.id = idx

    def track(self, frame):
        pass