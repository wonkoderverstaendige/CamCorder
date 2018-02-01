#!/usr/bin/env python
"""Record video from multiple cameras into one video file.

Crashes when requesting non-existing cameras to be captured. Just... don't.
"""

import cv2
import numpy as np
import time
from datetime import datetime

print('OpenCV version', cv2.__version__)

N_CAMERAS = 2
FPS = 15.
FOURCC = 'DIVX'

FONT = cv2.FONT_HERSHEY_PLAIN


def fmt_time(t):
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return "{h:02.0f}:{m:02.0f}:{s:06.3f}".format(h=h, m=m, s=s)


def main(video_out=None, fourcc=FOURCC, font=FONT, fps=FPS, record=False):
    t_start = time.time()
    captures = [cv2.VideoCapture(n) for n in range(N_CAMERAS)]
    captures.reverse()
    assert None not in captures

    writer = None

    while True:
        start = time.time()

        rvs, frames = zip(*[capture.read() for capture in captures])
        if all(rvs) and all([f is not None for f in frames]):
            # Merge individual frames along specified axis
            big_frame = np.concatenate(frames, axis=0)
            size = big_frame.shape[::-1][1:3]

            t_str = fmt_time(time.time() - t_start)
            ox = 4
            oy = 4
            font_scale = 1.5
            thickness = 1
            ts, bl = cv2.getTextSize(t_str, font, font_scale, thickness+2)

            cv2.rectangle(big_frame, (ox-thickness, size[1]-oy+thickness),
                          (ox+ts[0], size[1]-oy-ts[1]-thickness), (0, 0, 0), cv2.FILLED)

            cv2.putText(big_frame, t_str, (ox, size[1]-oy), font, font_scale, (255, 255, 255),
                        thickness, lineType=cv2.LINE_AA)

            # Write to disk
            start_write = time.time()
            if record:
                if writer is None:
                    if video_out is None:
                        video_out = "recordings/{:%Y%m%d_%H%M%S}.avi".format(datetime.now())
                    writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc(*fourcc),
                                             fps=15.0, frameSize=size, isColor=True)

                writer.write(big_frame)
            end_write = time.time()


            # Display merged frame
            start_disp = time.time()
            cv2.imshow('Merged view @ {}'.format(video_out), big_frame)
            end_disp = time.time()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Timing info
            end = time.time()
            elapsed = (end-start)*1000
            elapsed_write = (end_write - start_write)*1000
            elapsed_disp = (end_disp - start_disp)*1000
            print('{} - {:3.0f} ms ({:2.0f} ms write, {:1.0f} ms display), {:2.1f} fps'.format(
                fmt_time(time.time() - t_start), elapsed, elapsed_write, elapsed_disp, 1000/elapsed))
                
    [capture.release() for capture in captures]
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
