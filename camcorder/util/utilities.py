from math import ceil, sqrt

import cv2


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