import ctypes
import argparse
import logging
import threading
from queue import Queue
import multiprocessing as mp

import cv2
import numpy as np

from camcorder.util.defaults import *
from camcorder.util.utilities import buf_to_numpy, text_overlay
from camcorder.lib.grabber import Grabber
from camcorder.lib.writer import Writer
from camcorder.lib.tracker import Tracker

# shared buffer for transferring frames between threads/processes
SHARED_ARR = None


class HexTrack:
    def __init__(self, sources, shared_arr, frame_shape):
        threading.current_thread().name = 'HexTrack'

        # Control events
        self.ev_stop = threading.Event()
        self.ev_recording = threading.Event()
        self.ev_tracking = threading.Event()

        # List of video sources
        self.sources = sources

        # dummy
        self.denoising = False

        w, h, c = frame_shape

        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(self._shared_arr))
            self.frame = buf_to_numpy(self._shared_arr, shape=(h * len(sources), w, c))
        self.paused_frame = np.zeros_like(self.frame)

        # Frame queues for video file output
        self.queues = [Queue(maxsize=16) for _ in range(len(sources))]

        # Frame acquisition objects

        self.grabbers = [Grabber(source=self.sources[n], arr=self._shared_arr, out_queue=self.queues[n],
                                 trigger_event=self.ev_stop, idx=n) for n in range(len(sources))]

        # Video storage writers
        self.writers = [Writer(in_queue=self.queues[n], ev_alive=self.ev_stop, ev_recording=self.ev_recording, idx=n)
                        for n in range(len(sources))]

        # Online tracker
        self.trackers = [Tracker(idx=n) for n in range(len(sources))]

        # Scrolling frame on side
        self.node_frame = np.zeros((FRAME_HEIGHT, 200), dtype=np.uint8)
        self.node_frame[:20] = 255

        # Start up threads/processes
        for n in range(len(sources)):
            self.grabbers[n].start()
            self.writers[n].start()

        logging.debug('HexTrack initialization done!')

        self.paused = False

    def run(self):
        res = None
        while all([grabber.is_alive() for grabber in self.grabbers]):

            if self.paused:
                frame = self.paused_frame
            else:
                # Using copy prevents the image buffer to be overwritten by a new incoming frame
                # Question is, what is worse. Waiting with a lock until drawing is complete,
                # or the overhead of making a full frame copy.
                # TODO: Blit the frame here into an allocated display buffer
                frame = self.frame.copy()
                self.node_frame[2:] = self.node_frame[:-2]
                self.node_frame[:2] = 0
                node_updates = [tracker.track(frame) for tracker in self.trackers]
                for n, res in enumerate(node_updates):
                    if res is not None:
                        pass

            cv2.imshow('frame', frame)
            cv2.imshow('Node visits', self.node_frame)
            # What annoys a noisy oyster? Denoising noise annoys the noisy oyster!
            # This is for demonstrating a slow processing step not hindering the acquisition/writing threads
            if self.denoising and ALLOW_DUMMY_PROCESSING:
                t = cv2.getTickCount()
                dn = cv2.fastNlMeansDenoisingColored(self.frame, None, 6, 6, 5, 15)
                logging.debug((cv2.getTickCount() - t) / cv2.getTickFrequency())
                cv2.imshow('denoised', dn)

            key = cv2.waitKey(1)
            if key == ord('q'):
                self.ev_stop.set()
                logging.debug('Join request sent!')

                for grabber in self.grabbers:
                    grabber.join()
                logging.debug('All Grabbers joined!')

                for writer in self.writers:
                    writer.join()
                logging.debug('All Writers joined!')

                break

            elif key == ord('r'):
                if not self.ev_recording.is_set():
                    self.ev_recording.set()
                else:
                    self.ev_recording.clear()

            elif key == ord(' '):
                self.paused = not self.paused
                if self.paused:
                    self.paused_frame = self.frame.copy()

            elif key == ord('d'):
                self.denoising = not self.denoising


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sources', nargs='*', help='List of sources to read from', default=FRAME_SOURCES)
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')

    cli_args = parser.parse_args()

    if cli_args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - (%(threadName)-9s) %(message)s', )
    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - (%(threadName)-9s) %(message)s', )

    # Construct the shared array to fit all frames
    width = FRAME_WIDTH
    height = FRAME_HEIGHT
    colors = FRAME_COLORS
    fps = FRAME_FPS
    num_bytes = width * height * colors * len(cli_args.sources)

    SHARED_ARR = mp.Array(ctypes.c_ubyte, num_bytes)
    logging.debug('Created shared array: {}'.format(SHARED_ARR))

    mp.freeze_support()

    ht = HexTrack(sources=cli_args.sources, shared_arr=SHARED_ARR, frame_shape=(width, height, colors))
    ht.run()
