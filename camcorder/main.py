#!/usr/bin/env python
"""Record video from multiple cameras into one video file.

Crashes when requesting non-existing cameras to be captured. Just... don't.
"""

import cv2
import numpy as np
import time
from datetime import datetime

print('CamCorder with OpenCV v {}'.format(cv2.__version__))

NUM_CAMERAS = 2
FPS = 15.
FOURCC = 'DIVX'
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240

FONT = cv2.FONT_HERSHEY_PLAIN


def fmt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return "{h:02.0f}:{m:02.0f}:{s:06.3f}".format(h=h, m=m, s=s)


class CamCorder:
    def __init__(self, video_out=None, fourcc=FOURCC, font=FONT, fps=FPS, num_cameras=NUM_CAMERAS,
                 width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.num_cameras = num_cameras
        self.video_out = video_out
        self.fourcc = fourcc
        self.font = font
        self.fps = fps

        self.captures = [cv2.VideoCapture(n) for n in range(self.num_cameras)]
        self.captures.reverse()

        # set video size
        for capture in self.captures:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)

        assert None not in self.captures
        self.joint_frame = None
        self.capturing = True
        self.frame_size = None

        self.recording = False
        self.writer = None

        self.loop()

    def loop(self):
        self.t_start = time.time()
        while self.capturing:
            rvs, frames = zip(*[capture.read() for capture in self.captures])
            if all(rvs) and all([f is not None for f in frames]):
                # Merge individual frames along specified axis
                self.joint_frame = np.concatenate(frames, axis=0)
                self.frame_size = self.joint_frame.shape[::-1][1:3]
                self.add_overlay(self.joint_frame, time.time() - self.t_start)
            else:
                print('Capture unsuccessful')
                continue

            if self.recording:
                self.write(self.joint_frame)

            start_disp = time.time()
            cv2.imshow('Merged view @ {}'.format('CamCorder'), self.joint_frame)
            end_disp = time.time()

            kv = cv2.waitKey(1) & 0xFF
            if kv == ord('q'):
                break
            elif kv == ord('r'):
                if self.recording:
                    self.stop_recording()
                else:
                    self.start_recording()
                self.t_start = time.time()
            else:
                pass

            # # Timing info
            # end = time.time()
            # elapsed = (end-start)*1000
            # elapsed_write = (end_write - start_write)*1000
            # elapsed_disp = (end_disp - start_disp)*1000
            # print('{} - {:3.0f} ms ({:2.0f} ms write, {:1.0f} ms display), {:2.1f} fps'.format(
            #     fmt_time(time.time() - t_start), elapsed, elapsed_write, elapsed_disp, 1000/elapsed))
        self.close()

    def add_overlay(self, frame, t):
        t_str = fmt_time(t)
        ox, oy = 4, 4
        osx = 15
        thickness = 1
        font_scale = 1.5
        ts, bl = cv2.getTextSize(t_str, self.font, font_scale, thickness + 2)

        if not self.recording:
            bg, fg = (0, 0, 0), (255, 255, 255)
            radius = 0
        else :
            bg, fg = (255, 255, 255), (0, 0, 0)
            radius = 8

        cv2.rectangle(frame, (ox - thickness, self.frame_size[1] - oy + thickness),
                      (ox + ts[0] + 2*radius, self.frame_size[1] - oy - ts[1] - thickness), bg, cv2.FILLED)

        cv2.putText(frame, t_str, (ox, self.frame_size[1] - oy), self.font, font_scale, fg,
                    thickness, lineType=cv2.LINE_AA)

        if self.recording:
            cv2.circle(frame, (ox + ts[0] + radius, self.frame_size[1] - ts[1]//2 - oy), radius, (0, 0, 255), -1)

    def start_recording(self):
        video_out = "recordings/{:%Y%m%d_%H%M%S}.avi".format(datetime.now())
        self.writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc(*self.fourcc),
                                 fps=self.fps, frameSize=self.frame_size)
        self.recording = True

    def stop_recording(self):
        self.writer.release()
        self.writer = None
        self.recording = False

    def write(self, frame):
        if self.recording:
            start_write = time.time()
            self.writer.write(frame)
            end_write = time.time()

    def close(self):
        [capture.release() for capture in self.captures]
        cv2.destroyAllWindows()

if __name__ == '__main__':
    CamCorder()
