#!/usr/bin/env python
"""Record video from multiple cameras into one video file.

Crashes when requesting non-existing cameras to be captured. Just... don't.
"""

import csv
import time
from math import floor, ceil, sqrt
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

print('CamCorder with OpenCV v{}'.format(cv2.__version__))

FPS = 15.
ROTATE = False
FOURCC = 'DIVX'
VIDEO_EXT = 'avi'
VIDEO_WIDTH = 800
VIDEO_HEIGHT = 600
REVERSE_VIEW_ORDER = False
DRAW_OVERLAY = True

DEFAULT_SOURCES = [0, 1]
DEFAULT_OUTPUT_DIR = Path.home() / 'Videos' / 'hexmaze'

FONT = cv2.FONT_HERSHEY_PLAIN

if DRAW_OVERLAY:
    with open('overlay_vstack_rel.csv') as f:
        node_list = list(csv.reader(f))[1:]
else:
    node_list = []


def fmt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return "{h:02.0f}:{m:02.0f}:{s:06.3f}".format(h=h, m=m, s=s)


def text_overlay(frame, text, x=3, y=3, f_scale=1., color=None, origin='left', thickness=1):
    if color is None:
        if frame.ndim < 3:
            color = (255,)
        else:
            color = (255, 255, 255)
    color_bg = [1 for _ in color]
    outline_w = ceil(thickness + sqrt(2 * thickness + 1))

    f_h = int(13 * f_scale)
    x_ofs = x
    y_ofs = y + f_h

    lines = text.split('\n')

    for n, line in enumerate(lines):
        text_size, _ = cv2.getTextSize(line, fontFace=cv2.FONT_HERSHEY_PLAIN,
                                       fontScale=f_scale, thickness=outline_w)
        if origin == 'right':
            text_x = x_ofs - text_size[0]
        else:
            text_x = x_ofs

        # draw text outline
        cv2.putText(frame,
                    line, (text_x, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color_bg,
                    lineType=cv2.LINE_AA,
                    thickness=outline_w)

        # actual text
        cv2.putText(frame,
                    line, (text_x, y_ofs + n * f_h),
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=f_scale,
                    color=color,
                    lineType=cv2.LINE_AA,
                    thickness=thickness)


def generate_node_overlay(width, height, nodes):
    node_img = np.zeros((height, width, 3), np.uint8)

    for node in nodes:
        if int(node[6]):
            text_overlay(node_img, node[1], int(float(node[4]) * width), int(float(node[5]) * height),
                         thickness=2, f_scale=2.)

    node_img_gray = cv2.cvtColor(node_img, cv2.COLOR_BGR2GRAY)
    _, node_mask = cv2.threshold(node_img_gray, 1, 255, cv2.THRESH_BINARY_INV)
    return node_mask, node_img


class CamCorder:
    def __init__(self, sources, out_path, fourcc=FOURCC, font=FONT, fps=FPS,
                 width=VIDEO_WIDTH, height=VIDEO_HEIGHT):
        self.sources = sources
        self.out_path = out_path
        self.video_out = None
        self.fourcc = fourcc
        self.font = font
        self.fps = fps

        captures = []
        for src in sources:
            try:
                captures.append(int(src))
                is_file = False
            except ValueError:
                src = Path(src)
                if not src.exists() and src.is_file():
                    raise FileNotFoundError("Can't find requested file: {}".format(str(src)))
                is_file = True
                captures.append(str(src))

        self.captures = [cv2.VideoCapture(src) for src in captures]

        # reverse order
        if REVERSE_VIEW_ORDER:
            self.captures.reverse()

        # set video source size and fps (only affects camera sources)
        for capture in self.captures:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_FPS, fps)

        assert None not in self.captures

        self.joint_frame = np.zeros((width, height * 2, 3))
        self.capturing = True
        self.frame_size = None

        self.recording = False
        self.writer = None

        self.t_start = None

        self.node_mask, self.node_img = generate_node_overlay(width, height * 2, node_list)
        self.loop()

    def loop(self):
        print("Entering acquisition loop")
        self.t_start = time.time()
        while self.capturing:
            t_acq = time.time()

            # Get frames from all sources
            rvs, frames = zip(*[capture.read() for capture in self.captures])
            if all(rvs) and all([f is not None for f in frames]):
                # Merge individual frames along specified axis
                self.joint_frame = np.concatenate(frames, axis=0)
                # if ROTATE:
                #     # Weird opencv issue with drawing into frame if not copied after rotation
                #     self.joint_frame = np.rot90(self.joint_frame).copy()
                self.frame_size = self.joint_frame.shape[::-1][1:3]
                self.add_overlay(self.joint_frame, time.time() - self.t_start)
            else:
                print('Capture unsuccessful')
                self.capturing = False
                continue

            if self.recording:
                self.write(self.joint_frame)

            # self.add_node_overlay(self.joint_frame, node_list)

            # start_disp = time.time()
            if not DRAW_OVERLAY:
                cv2.imshow('Merged view @ {}'.format('CamCorder'), self.joint_frame)
            else:
                cv2.imshow('Merged view @ {}'.format('CamCorder'),
                           np.bitwise_and(self.node_mask[:, :, None], self.joint_frame) + self.node_img)
            # end_disp = time.time()

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

            # Timing info
            t_acq_end = time.time()
            elapsed = (t_acq_end - t_acq)
            print("\r{:.0f} ms/frame, {:.1f} fps   ".format(elapsed * 1000, 1. / elapsed), end='', flush=True)
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
        font_scale = 1.2

        ts, bl = cv2.getTextSize(t_str, self.font, font_scale, thickness + 2)

        if not self.recording:
            bg, fg = (0, 0, 0), (255, 255, 255)
            radius = 0
        else:
            bg, fg = (255, 255, 255), (0, 0, 0)
            radius = 8

        cv2.rectangle(frame, (ox - thickness, self.frame_size[1] - oy + thickness),
                      (ox + ts[0] + 2 * radius, self.frame_size[1] - oy - ts[1] - thickness), bg, cv2.FILLED)

        cv2.putText(frame, t_str, (ox, self.frame_size[1] - oy), self.font, font_scale, fg,
                    thickness, lineType=cv2.LINE_AA)

        if self.recording:
            cv2.circle(frame, (ox + ts[0] + radius, self.frame_size[1] - ts[1] // 2 - oy), radius, (0, 0, 255), -1)

    def add_node_overlay(self, frame, nodes):
        # Draw node names
        for node in nodes:
            if int(node[6]):
                text_overlay(frame, node[1], int(float(node[4]) * VIDEO_WIDTH), int(float(node[5]) * 2 * VIDEO_HEIGHT),
                             thickness=2, f_scale=2.)

    # def add_node_overlay_masked(self, frame):
    #     # Draw node names
    #     for node in nodes:
    #         if int(node[6]):
    #             text_overlay(frame, node[1], int(float(node[4]) * VIDEO_WIDTH), int(float(node[5]) * 2 * VIDEO_HEIGHT),
    #                          thickness=2, f_scale=2.)

    def start_recording(self):
        print('Writer fps: ', self.fps)
        video_out = str(self.out_path / "{:%Y%m%d_%H%M%S}.{}".format(datetime.now(), VIDEO_EXT))
        self.writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc(*self.fourcc),
                                      fps=self.fps, frameSize=self.frame_size)
        self.recording = True

    def stop_recording(self):
        self.writer.release()
        self.writer = None
        self.recording = False

    def write(self, frame):
        if self.recording:
            # start_write = time.time()
            self.writer.write(frame)
            # end_write = time.time()

    def close(self):
        [capture.release() for capture in self.captures]
        cv2.destroyAllWindows()
        print('\n Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', help='Video output location.', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('-i', '--input', help='Video source(s)', nargs='*', default=DEFAULT_SOURCES)

    cli_args = parser.parse_args()

    dst_path = Path(cli_args.output)

    if not dst_path.exists():
        dst_path.mkdir()
        print('Created output directory {}'.format(dst_path))

    if not dst_path.is_dir():
        raise FileExistsError("Can't create target directory. Aborting.")

    dst_path = dst_path.resolve()

    CamCorder(sources=cli_args.input, out_path=dst_path)
