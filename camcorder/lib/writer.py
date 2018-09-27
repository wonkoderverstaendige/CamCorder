import logging
import threading
import time
from pathlib import Path
from queue import Empty

import cv2

from camcorder.util.defaults import *

class Writer(threading.Thread):
    def __init__(self, in_queue, ev_alive, ev_recording, idx=0):
        super().__init__()
        self.id = idx
        self.name = 'Writer ' + str(self.id)
        self.writer = None

        self.n_frames = 0
        self.width = None
        self.height = None
        self.frame = None

        self.in_queue = in_queue
        self._ev_stop = ev_alive
        self._ev_recording = ev_recording

        self.recording = False

        self.codec = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # cv2.VideoWriter_fourcc(*'MP4V')
        self.container = VIDEO_CONTAINER
        self.video_fname = str(Path.home() / "Videos/hextrack/{}_cam_{}") + self.container

        logging.debug('Writer initialization done!')

    def run(self):
        logging.debug('Starting loop in {}!'.format(self.name))
        while not self._ev_stop.is_set():
            try:
                self.frame = self.in_queue.get(timeout=.5)
            except Empty:
                continue

            h, w = self.frame.img.shape[:2]
            if self.recording != self._ev_recording.is_set():
                self.recording = self._ev_recording.is_set()
                if self.recording:
                    ts = time.strftime("%d-%b-%y_%H-%M-%S", time.localtime(time.time()))
                    fname = self.video_fname.format(ts, self.id)
                    logging.debug('Starting Recording to {}'.format(fname))

                    # Video output object
                    self.writer = cv2.VideoWriter(fname, self.codec, 15., (w, h))
                else:
                    logging.debug('Stopping Recording')
                    self.writer.release()

            if self.recording:
                self.writer.write(self.frame.img)

        logging.debug('Stopping loop in {}!'.format(self.name))
        if self.writer is not None:
            self.writer.release()
