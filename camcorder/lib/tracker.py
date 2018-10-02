import cv2
import math
import numpy as np
import logging
from collections import deque

from camcorder.util.defaults import *

MIN_MOUSE_AREA = 50
MIN_DIST_TO_NODE = 100

THICKNESS_MINOR_CONTOUR = 1
THICKNESS_MAJOR_CONTOUR = 1
DRAW_MINOR_CONTOURS = False
DRAW_MAJOR_CONTOURS = True

TRAIL_LENGTH = 100
DRAW_TRAIL = True

KERNEL_3 = np.ones((3, 3), np.uint8)

nodes = [NODES_A, NODES_B]
leds = [LED_A, LED_B]


def centroid(cnt):
    m = cv2.moments(cnt)
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    return cx, cy


def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Tracker:
    def __init__(self, idx=0, thresh_mask=100, thresh_detect=35, thresh_led=70):
        super().__init__()
        self.id = idx
        self.n_frames = 0
        self.thresh_mask = thresh_mask
        self.thresh_detect = 255 - thresh_detect
        self.thresh_led = thresh_led

        self.mask_frame = np.zeros((600, 800), np.uint8)
        self.have_mask = False

        self.nodes = nodes[self.id]
        self.results = deque(maxlen=TRAIL_LENGTH)
        self.last_node = None

        self.led_pos = leds[self.id]
        self.led_state = False

    def track(self, frame):
        node_updated = False
        img = frame[self.id * 600:(self.id + 1) * 600, :]
        foi = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # It takes time to fire up the cameras, so first frames might be zeros.
        # Check until we have a mask
        if not self.have_mask and np.sum(np.sum(foi)):
            logging.debug('Creating mask')
            _, mask = cv2.threshold(foi, self.thresh_mask, 255, cv2.THRESH_BINARY)
            self.mask_frame = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_3)
            self.have_mask = True

        masked = cv2.bitwise_not(foi) * (self.mask_frame // 255)
        masked = cv2.morphologyEx(masked, cv2.MORPH_OPEN, KERNEL_3)

        _, thresh = cv2.threshold(masked, self.thresh_detect, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL_3)

        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find largest contour
        largest_cnt, largest_area = None, 0
        sum_area = 0
        for cnt in contours:
            area = int(cv2.contourArea(cnt))
            if area > MIN_MOUSE_AREA:
                sum_area += area
                if area > largest_area:
                    largest_area = area
                    largest_cnt = cnt

        if DRAW_MINOR_CONTOURS:
            cv2.drawContours(img, contours, -1, (150, 150, 0), THICKNESS_MINOR_CONTOUR)

        closest_node = None
        closest_distance = 1e12

        if largest_cnt is None:
            self.results.appendleft(None)
        else:
            # center coordinates of contour
            cx, cy = centroid(largest_cnt)
            self.results.appendleft((cx, cy))

            # draw largest contour and contour label
            if DRAW_MAJOR_CONTOURS:
                cv2.drawContours(img, [largest_cnt], 0, (0, 0, 255), THICKNESS_MAJOR_CONTOUR)
                # overlay(self.frame['raw'],
                #         text='{}, {}\nA: {}'.format(cx, cy, largest_area),
                #         x=(min(cx + 15, 700)),
                #         y=cy + 15)

            cv2.drawMarker(img=img, position=centroid(largest_cnt), color=(0, 255, 0))
            # cv2.circle(self.frame['raw'], (cx, cy), 3, color=(255, 255, 255))

            # Find closest node
            for node_id, node in self.nodes.items():
                dist = distance(cx, cy, node['x'], node['y'])
                if dist < closest_distance and dist < MIN_DIST_TO_NODE:
                    closest_distance = dist
                    closest_node = node_id

            if self.last_node != closest_node:
                self.last_node = closest_node
                node_updated = True
                logging.info('Tracker {}: {} {}'.format(self.id, '    ' * self.id, self.last_node))

        # Label nodes
        for node_id, node in self.nodes.items():
            color = (255, 0, 0) if node_id == closest_node else (255, 255, 255)
            cv2.circle(img, (node['x'], node['y']), MIN_DIST_TO_NODE // 2, color)

            # overlay(self.frame['raw'], text=str(node_id), color=color,
            #         x=node['x'] - self.x, y=node['y'] - self.y, f_scale=2.)

        # Draw the trail
        points = self.results
        if DRAW_TRAIL and len(points) > 1:
            for p_idx in range(len(points) - 1):
                try:
                    x1, y1 = map(int, points[p_idx])
                    x2, y2 = map(int, points[p_idx + 1])
                except (ValueError, TypeError):
                    pass
                else:
                    cv2.line(img, (x1, y1), (x2, y2), color=(255, 255, 255))

        # Detect LED state
        self.led_state = foi[self.led_pos[1], self.led_pos[0]] > self.thresh_led

        self.n_frames += 1
        if node_updated:
            return self.led_state, self.last_node
        else:
            return self.led_state, None
