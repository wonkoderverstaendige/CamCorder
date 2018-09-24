import ctypes
import argparse
import logging
import threading
from queue import Queue
import multiprocessing as mp

import cv2
import numpy as np

from camcorder.util.defaults import FRAME_WIDTH, FRAME_HEIGHT, FRAME_COLORS, FRAME_FPS, FRAME_SOURCES
from camcorder.util.utilities import buf_to_numpy
from camcorder.lib.grabber import Grabber
from camcorder.lib.writer import Writer
from camcorder.lib.tracker import Tracker

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - (%(threadName)-9s) %(message)s', )

# shared buffer for transferring frames between threads/processes
shared_arr = None


class HexTrack:
    def __init__(self, sources, shared_arr, frame_shape):
        threading.current_thread().name = 'HexTrack'

        # Control events
        self.ev_stop = threading.Event()
        self.ev_recording = threading.Event()
        self.ev_tracking = threading.Event()

        # List of video sources
        self.sources = sources

        self.denoising = False

        w, h, c = frame_shape

        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(self._shared_arr))
            self.frame = buf_to_numpy(self._shared_arr, shape=(h * len(sources), w, c))
        self.paused_frame = np.zeros_like(self.frame)

        # Frame queues for video file output
        self.queues = [Queue(maxsize=16) for n in range(len(sources))]

        # Frame acquisition objects

        self.grabbers = [Grabber(source=self.sources[n], arr=self._shared_arr, out_queue=self.queues[n],
                               trigger_event=self.ev_stop, idx=n) for n in range(len(sources))]

        # Video storage writers
        self.writers = [Writer(in_queue=self.queues[n], ev_alive=self.ev_stop, ev_recording=self.ev_recording, idx=n) for n in range(len(sources))]


        # Online tracker
        self.trackers = [Tracker(idx=n) for n in range(len(sources))]

        # Start up threads/processes
        for n in range(len(sources)):
            self.grabbers[n].start()
            self.writers[n].start()

        logging.debug('HexTrack initialization done!')

        self.paused = False

    def run(self):
        while all([grabber.is_alive() for grabber in self.grabbers]):

            if self.paused:
                frame = self.paused_frame
            else:
                frame = self.frame

            cv2.imshow('frame', frame)

            # What annoys a noisy oyster? Denoising noise annoys the noisy oyster!
            # This is for demonstrating a slow processing step not hindering the acquisition/writing threads
            if self.denoising:
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

    # Construct the shared array to fit all frames
    sources = FRAME_SOURCES
    width = FRAME_WIDTH
    height = FRAME_HEIGHT
    colors = FRAME_COLORS
    fps = FRAME_FPS
    num_bytes = width * height * colors * len(sources)

    shared_arr = mp.Array(ctypes.c_ubyte, num_bytes)
    logging.debug('Created shared array: {}'.format(shared_arr))

    mp.freeze_support()

    ht = HexTrack(sources=sources, shared_arr=shared_arr, frame_shape=(width, height, colors))
    ht.run()
