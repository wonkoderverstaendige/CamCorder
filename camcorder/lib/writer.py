import cv2
import csv
import time
import logging
import threading
from queue import Empty
from pathlib import Path

from camcorder.util.defaults import *


class Writer(threading.Thread):
    def __init__(self, in_queue, ev_alive, ev_recording, ev_trial_active, idx=0):
        super().__init__()
        self.id = idx
        self.name = 'Writer ' + str(self.id)
        self.writer = None
        self.logger = None

        self.n_frames = 0
        self.width = None
        self.height = None
        self.frame = None

        self.in_queue = in_queue
        self._ev_stop = ev_alive
        self._ev_recording = ev_recording
        self._ev_trial_active = ev_trial_active

        self.recording = False

        self.codec = cv2.VideoWriter_fourcc(*VIDEO_CODEC)  # cv2.VideoWriter_fourcc(*'MP4V')
        self.video_fname = Path.home() / "Videos/hextrack/{}_cam_{}"
        self.video_fname = str(self.video_fname.as_posix())

        logging.debug('Writer initialization done!')

    def start_recording(self):
        h, w = self.frame.img.shape[:2]
        ts = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time()))
        fname_base = self.video_fname.format(ts, self.id)

        # Video output object
        path_video = fname_base + VIDEO_CONTAINER
        logging.debug('Starting Recording to {}'.format(path_video))
        self.writer = cv2.VideoWriter(path_video, self.codec, 15., (w, h))

        # Frame metadata logger output
        path_log = fname_base + '.csv'
        try:
            self.logger = csv.writer(open(path_log, 'w', newline=''))
        except FileNotFoundError:
            logging.error('Failed to open log file at {}'.format(path_log))
            self.stop_recording()

        self.logger.writerow(['index', 'bool_trial', 'timestamp', 'tickstamp'])
        self.recording = True

    def stop_recording(self):
        if self.recording:
            logging.debug('Stopping Recording')

        self.recording = False
        self._ev_recording.clear()
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        self.logger = None


    def run(self):
        logging.debug('Starting loop in {}!'.format(self.name))
        try:
            while not self._ev_stop.is_set():
                try:
                    self.frame = self.in_queue.get(timeout=.5)
                except Empty:
                    continue

                rec = self._ev_recording.is_set()
                if self.recording != rec:
                    if rec:
                        self.start_recording()
                    else:
                        self.stop_recording()

                if self.recording:
                    if self.writer is None:
                        logging.error('Attempted to write to failed Writer!')
                        self.stop_recording()
                    else:
                        self.writer.write(self.frame.img)
                        metadata = [self.frame.index, int(self._ev_trial_active.is_set()),
                                    self.frame.time_text, self.frame.tickstamp]
                        self.logger.writerow(metadata)

            logging.debug('Stopping loop in {}!'.format(self.name))
        except:
            raise
        finally:
            self.stop_recording()
