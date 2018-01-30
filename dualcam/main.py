import cv2
import numpy as np
import time
from datetime import datetime

print('OpenCV version', cv2.__version__)

N_CAMERAS = 2
FONT = cv2.FONT_HERSHEY_PLAIN

def fmt_time(s, minimal=False):
    """
    Args:
        s: time in seconds (float for fractional)
        minimal: Flag, if true, only return strings for times > 0, leave rest outs
    Returns: String formatted 99h 59min 59.9s, where elements < 1 are left out optionally.
    """
    ms = s - int(s)
    s = int(s)
    if s < 60 and minimal:
        return "{s:02.3f}s".format(s=s + ms)

    m, s = divmod(s, 60)
    if m < 60 and minimal:
        return "{m:02d}min {s:02.3f}s".format(m=m, s=s + ms)

    h, m = divmod(m, 60)
    return "{h:02d}:{m:02d}:{s:02.3f}".format(h=h, m=m, s=s + ms)


def main(video_out=None, fourcc='X264', font=FONT):
    if video_out is None:
        video_out = "{:%Y%m%d_%H%M%S}.avi".format(datetime.now())

    t_start = time.time()
    captures = [cv2.VideoCapture(n) for n in range(1, -1, -1)]
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

            if writer is None:
                print(size)
                writer = cv2.VideoWriter(video_out, fourcc=cv2.VideoWriter_fourcc(*fourcc),
                                         fps=15.0, frameSize=size, isColor=True)

            # Write to disk
            start_write = time.time()
            writer.write(big_frame)
            end_write = time.time()

            # Display merged frame
            start_disp = time.time()
            cv2.imshow('Merged view @ {}'.format(video_out), big_frame)
            end_disp = time.time()

            # Timing info
            end = time.time()
            elapsed = (end-start)*1000
            elapsed_write = (end_write - start_write)*1000
            elapsed_disp = (end_disp - start_disp)*1000
            print('{} - {:3.0f} ms ({:2.0f} ms write, {:1.0f} ms display), {:2.1f} fps'.format(
                fmt_time(time.time() - t_start), elapsed, elapsed_write, elapsed_disp, 1000/elapsed))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    [capture.release() for capture in captures]
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()