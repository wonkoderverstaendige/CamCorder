import cv2

N_FRAMES_LOG_WINDOW = 100
N_FRAMES_FPS_WINDOW = 10

FRAME_WIDTH = 800
FRAME_HEIGHT = 600
FRAME_COLORS = 3
FRAME_FPS = 15.
FRAME_SOURCES = [0, 1]
NODE_FRAME_WIDTH = 105
SYNC_FRAME_WIDTH = 5

NODE_FRAME_BG_ACTIVE = (75, 75, 75)
NODE_FRAME_BG_INACTIVE = 0

NODE_FRAME_FG_ACTIVE = (255, 255, 255)
NODE_FRAME_FG_INACTIVE = (100, 100, 100)

FRAME_ADD_TICKSTAMP = False
FRAME_ADD_TIMESTAMP = False

VIDEO_CONTAINER = '.avi'
VIDEO_CODEC = 'MP42'

FONT = cv2.FONT_HERSHEY_PLAIN

PLAYBACK_SPEEDUP = 1

ALLOW_DUMMY_PROCESSING = False

NODES_A = {1: {'use': None, 'x': 84, 'y': 75},
           2: {'use': None, 'x': 227, 'y': 17},
           3: {'use': None, 'x': 357, 'y': 99},
           4: {'use': None, 'x': 499, 'y': 43},
           5: {'use': None, 'x': 631, 'y': 125},
           6: {'use': None, 'x': 54, 'y': 226},
           7: {'use': None, 'x': 339, 'y': 257},
           8: {'use': None, 'x': 624, 'y': 281},
           9: {'use': None, 'x': 186, 'y': 316},
           10: {'use': None, 'x': 476, 'y': 347},
           11: {'use': None, 'x': 765, 'y': 374},
           12: {'use': None, 'x': 162, 'y': 491},
           13: {'use': None, 'x': 463, 'y': 519},
           14: {'use': None, 'x': 762, 'y': 549},
           15: {'use': None, 'x': 314, 'y': 573},
           16: {'use': None, 'x': 625, 'y': 592}
           }

NODES_B = {
           6: {'use': None, 'x': 186, 'y': 11},
           7: {'use': None, 'x': 448, 'y': 15},
           8: {'use': None, 'x': 718, 'y': 25},
           9: {'use': None, 'x': 319, 'y': 72},
           10: {'use': None, 'x': 591, 'y': 93},
           12: {'use': None, 'x': 306, 'y': 224},
           13: {'use': None, 'x': 589, 'y': 243},
           15: {'use': None, 'x': 444, 'y': 312},
           16: {'use': None, 'x': 730, 'y': 332},
           17: {'use': None, 'x': 47, 'y': 57},
           18: {'use': None, 'x': 21, 'y': 206},
           19: {'use': None, 'x': 157, 'y': 296},
           20: {'use': None, 'x': 135, 'y': 462},
           21: {'use': None, 'x': 277, 'y': 558},
           22: {'use': None, 'x': 435, 'y': 482},
           23: {'use': None, 'x': 580, 'y': 581},
           24: {'use': None, 'x': 730, 'y': 500}}

LED_A = (376, 445)

LED_B = (539, 116)