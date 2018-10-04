import cv2
import numpy as np
from math import ceil, sqrt

from camcorder.util.defaults import *


def extract_metadata(arr):
    """Extract metadata from a small patch of pixels"""
    metadata = {}
    rows = arr.shape[0]
    for row in range(rows):
        try:
            s = arr[row].reshape(-1)
            g_id = int(s[0])
            if np.sum(s[1:7]):
                key = s[1:7].tostring().decode('ascii').strip()
                metadata[key] = (g_id, s[7:].view(np.uint64))
        except UnicodeDecodeError:
            pass
    return metadata


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


def buf_to_numpy(buf, shape, count=-1, offset=0):
    """Return numpy object from a raw buffer, e.g. multiprocessing Array"""
    return np.frombuffer(buf.get_obj(), dtype=np.ubyte, count=count, offset=offset).reshape(shape)


def fmt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return "{h:02.0f}:{m:02.0f}:{s:06.3f}".format(h=h, m=m, s=s)
