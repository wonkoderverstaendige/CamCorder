import time
import logging
import threading
from queue import Full
from collections import deque

import cv2
import numpy as np

from camcorder.util.defaults import *
from camcorder.util.utilities import buf_to_numpy


class Frame:
    def __init__(self, index, img, source_type, timestamp=None, tickstamp=None, add_stamps=True):
        """Container class for frames. Holds additional metadata aside from the
        actual image information."""
        self.img = img

        self.index = index
        self.source_type = source_type
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.tickstamp = tickstamp if tickstamp is not None else \
            int((1000 * cv2.getTickCount()) / cv2.getTickFrequency())

        time_text = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime(self.timestamp))
        ms = "{0:03d}".format(int((self.timestamp - int(self.timestamp)) * 1000))
        self.time_text = ".".join([time_text, ms])

        # Add timestamp to image if from a live source
        if add_stamps or FORCE_TIMESTAMPS:
            self.add_stamps()

    @property
    def width(self):
        return self.img.shape[0]

    @property
    def height(self):
        return self.img.shape[1]

    @property
    def shape(self):
        return self.img.shape

    def add_stamps(self):
        """Add tick- and timestamp to the unused section of the metadata frame.
        TODO: This should happen in Grabber, not the tracker, to record this in the video, too.
        """
        ty = self.img.shape[0] - 5
        tx = self.width - 143
        thickness = 1
        font_scale = 1.0
        bg, fg = (0, 0, 0), (255, 255, 255)

        ms = "{0:03d}".format(int((self.timestamp - int(self.timestamp)) * 1000))
        ts = time.strftime("%H:%M:%S.{}  %d.%m.%Y", time.localtime(self.timestamp)).format(ms)

        t_str = '{}  {}'.format(int(self.tickstamp), ts)

        cv2.putText(self.img, t_str, (tx, ty), FONT, font_scale, fg,
                    thickness, lineType=METADATA_LINE_TYPE)


class Grabber(threading.Thread):
    def __init__(self, cfg, source, arr, out_queue, trigger_event, idx=0):
        super().__init__()
        self.id = idx
        self.cfg = cfg

        self.name = 'Grabber ' + str(self.id)
        try:
            self.source = int(source)
        except ValueError:
            self.source = source

        self.is_live = not isinstance(self.source, str)

        self.n_frames = 0
        self.capture = None
        self.frame = None

        self.width = cfg['frame_width']
        self.height = cfg['frame_height']
        self.colors = cfg['frame_colors']

        shape = (self.height + FRAME_METADATA_H, self.width, self.colors)
        num_bytes = int(np.prod(shape))

        # Attach to shared buffer
        with arr.get_lock():
            self._shared_arr = arr
            logging.debug('Grabber shared array: {}'.format(arr))
            self._fresh_frame = buf_to_numpy(arr, shape=shape, offset=self.id * num_bytes, count=num_bytes)
            logging.debug('Numpy shared buffer at {}'.format(hex(self._fresh_frame.ctypes.data)))

        self._write_queue = out_queue
        self._ev_terminate = trigger_event
        self._avg_fps = cfg['frame_fps']
        self._t_loop = deque(maxlen=N_FRAMES_FPS_LOG)

        logging.debug('Grabber initialization done!')

    def run(self):
        logging.debug('Starting loop in {} with source {}'.format(self.name, self.source))
        self.capture = cv2.VideoCapture(self.source)

        # Request source to have
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # MAXIMUM fps. This is more a recommendation, most cameras don't listen to this.
        self.capture.set(cv2.CAP_PROP_FPS, self.cfg['frame_fps'])

        t0 = cv2.getTickCount()
        while not self._ev_terminate.is_set():
            rt, frame = self.capture.read()
            if not rt:
                continue

            # Make space for the metadata bar at the bottom of each frame
            frame.resize((frame.shape[0] + FRAME_METADATA_H, frame.shape[1], frame.shape[2]))
            self.frame = Frame(self.n_frames, frame, 'Grabber', add_stamps=self.is_live)

            # Send frames to attached threads/processes
            self.relay_frames()

            # Slow down "replay" if the image source is a video file to emulate real time replay
            if not self.is_live:
                time.sleep(1 / self.capture.get(cv2.CAP_PROP_FPS) / PLAYBACK_SPEEDUP)
            self.n_frames += 1

            self._t_loop.appendleft((cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000)
            t0 = cv2.getTickCount()

            # Every now and then show fps
            # if not self.n_frames % N_FRAMES_FPS_LOG:
            #     avg_fps = 1 / (sum(self._t_loop) / len(self._t_loop))
            #     logging.debug(
            #         'Grabbing frame {}... {}, avg. {:.1f} fps'.format(self.n_frames, 'OK' if rt else 'FAIL', avg_fps))

        logging.debug('Stopping loop in {}!'.format(self.name))

    def embed_metadata(self, row, label, data):
        """Embed metadata into pixels. Note: Data has to be int64-able.
        """
        line = np.zeros(FRAME_METADATA_BYTE, dtype=np.uint8)
        line[0] = np.array([self.id], dtype=np.uint8)
        line[1:7] = np.fromstring('{:<6s}'.format(label), dtype=np.uint8)
        line[7:] = np.array([data], dtype=np.uint64).view(np.uint8)
        self._fresh_frame[-FRAME_METADATA_H + row:-FRAME_METADATA_H + row + 1, -FRAME_METADATA_BYTE // 3:] = line.reshape(1, -1, 3)

    def relay_frames(self):
        """Forward acquired image to entities downstream via queues or shared array.
        """
        try:
            self._write_queue.put(self.frame, timeout=.5)
        except Full:
            logging.warning('Dropped frame {}'.format(self.frame.index))

        # FPS display
        if len(self._t_loop):
            fps_str = 'G={:.1f}fps'.format(1000 / (sum(self._t_loop) / len(self._t_loop)))
        else:
            fps_str = 'G=??.?fps'
        cv2.putText(self.frame.img, fps_str, (270, self.frame.img.shape[0] - 5), FONT, 1.0,
                    (255, 255, 255), lineType=cv2.LINE_AA)

        # Forward frame for tracking and display
        # NOTE: [:] indicates to reuse the buffer
        with self._shared_arr.get_lock():
            self._fresh_frame[:] = self.frame.img

            self.embed_metadata(row=0, label='index', data=self.frame.index)
            self.embed_metadata(row=1, label='tickst', data=self.frame.tickstamp)
            self.embed_metadata(row=2, label='timest', data=int(self.frame.timestamp))
