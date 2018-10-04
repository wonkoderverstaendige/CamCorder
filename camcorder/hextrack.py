#!/usr/bin/env python
import ctypes
import time
import argparse
import logging
import threading
from pathlib import Path
from queue import Queue
import multiprocessing as mp
from collections import deque

import numpy as np

from camcorder.util.defaults import *
from camcorder.util.utilities import buf_to_numpy, fmt_time
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
        self.ev_trial_active = threading.Event()
        self.t_phase = cv2.getTickCount()

        self._loop_times = deque(maxlen=30)

        # List of video sources
        self.sources = sources

        # dummy
        self.denoising = False

        w, h, c = frame_shape

        # Shared array population
        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(self._shared_arr))
            self.frame = buf_to_numpy(self._shared_arr, shape=(h * len(sources), w, c))
        self.paused_frame = np.zeros_like(self.frame)

        # Allocate scrolling frame with node visits
        self.disp_frame = np.zeros((h * len(sources), w + NODE_FRAME_WIDTH, 3), dtype=np.uint8)
        self.disp_frame[:1] = NODE_FRAME_FG_INACTIVE

        # Frame queues for video file output
        self.queues = [Queue(maxsize=16) for _ in range(len(sources))]

        # Frame acquisition objects

        self.grabbers = [Grabber(source=self.sources[n], arr=self._shared_arr, out_queue=self.queues[n],
                                 trigger_event=self.ev_stop, idx=n) for n in range(len(sources))]

        # Video storage writers
        self.writers = [Writer(in_queue=self.queues[n], ev_alive=self.ev_stop, ev_recording=self.ev_recording,
                               ev_trial_active=self.ev_trial_active, idx=n) for n in range(len(sources))]

        # Online tracker
        self.trackers = [Tracker(idx=n) for n in range(len(sources))]

        # Start up threads/processes
        for n in range(len(sources)):
            self.grabbers[n].start()
            self.writers[n].start()

        cv2.namedWindow('HexTrack', cv2.WINDOW_AUTOSIZE)

        logging.debug('HexTrack initialization done!')

        self.paused = False

    def loop(self):
        frame_idx = 0
        t0 = cv2.getTickCount()
        while all([grabber.is_alive() for grabber in self.grabbers]):
            if self.paused:
                frame = self.paused_frame
            else:
                # Using copy prevents the image buffer to be overwritten by a new incoming frame
                # Question is, what is worse - waiting with a lock until drawing is complete,
                # or the overhead of making a full frame copy.
                # TODO: Blit the frame here into an allocated display buffer
                with self._shared_arr.get_lock():
                    frame = self.frame.copy()

                for tracker in self.trackers:
                    tracker.apply(frame)
                    tracker.annotate()

                frame_idx += 1
                delta = NODE_FRAME_STEP_SCROLL

                self.disp_frame[:, :FRAME_WIDTH] = frame

                trial_active = self.ev_trial_active.is_set()
                fg_col = NODE_FRAME_FG_ACTIVE if trial_active else NODE_FRAME_FG_INACTIVE
                bg_col = NODE_FRAME_BG_ACTIVE if trial_active else NODE_FRAME_BG_INACTIVE

                self.disp_frame[delta:, FRAME_WIDTH:] = self.disp_frame[:-delta, FRAME_WIDTH:]
                self.disp_frame[:delta, FRAME_WIDTH:] = bg_col

                self.disp_frame[:delta, FRAME_WIDTH + NODE_FRAME_WIDTH - SYNC_FRAME_WIDTH:] = NODE_FRAME_BG_INACTIVE

                for tracker in self.trackers:
                    # Draw colored bands for detected LEDs
                    if tracker.led_state:
                        col = [25, 25, 25]
                        col[tracker.id + 1] = 127
                        self.disp_frame[:delta, FRAME_WIDTH + NODE_FRAME_WIDTH - SYNC_FRAME_WIDTH:] = \
                            self.disp_frame[:delta, FRAME_WIDTH + NODE_FRAME_WIDTH - SYNC_FRAME_WIDTH:] + col

                    if not tracker.node_updated_presented:
                        tracker.node_updated_presented = True
                        # Put most recent node visit into the scrolling node frame
                        if tracker.last_node is not None:
                            cv2.putText(self.disp_frame, '{: >2d}'.format(tracker.last_node),
                                        (FRAME_WIDTH + (5 + tracker.id * 50), 20),
                                        FONT, NODE_FRAME_FONT_SIZE, fg_col, thickness=NODE_FRAME_FONT_WEIGHT)

                # Timestamp overlay
                self.add_overlay(self.disp_frame, (cv2.getTickCount() - self.t_phase) / cv2.getTickFrequency())

            cv2.imshow('HexTrack', self.disp_frame)

            # What annoys a noisy oyster? Denoising noise annoys the noisy oyster!
            # This is for demonstrating a slow processing step not hindering the acquisition/writing threads
            if self.denoising and ALLOW_DUMMY_PROCESSING:
                t = cv2.getTickCount()
                dn = cv2.fastNlMeansDenoisingColored(self.frame, None, 6, 6, 5, 15)
                logging.debug((cv2.getTickCount() - t) / cv2.getTickFrequency())
                cv2.imshow('denoised', dn)

            # Event loop call
            key = cv2.waitKey(25)

            # Process Keypress Events
            if key == ord('q'):
                self.ev_stop.set()
                logging.debug('Join request sent!')

                # Shut down Grabbers
                for grabber in self.grabbers:
                    grabber.join()
                logging.debug('All Grabbers joined!')

                # Shut down Writers
                for writer in self.writers:
                    writer.join()
                logging.debug('All Writers joined!')

                break

            elif key == ord('r'):
                # Start or stop recording
                self.t_phase = cv2.getTickCount()
                if not self.ev_recording.is_set():
                    self.ev_recording.set()
                else:
                    self.ev_recording.clear()

            elif key == ord(' '):
                # Pause display (not acquisition!)
                self.paused = not self.paused
                if self.paused:
                    self.paused_frame = self.frame.copy()

            elif key == ord('d'):
                # Enable dummy processing to slow down main loop
                # demonstrates the backend continuously grabbing and
                # writing frames even of display/tracking is slow
                self.denoising = not self.denoising

            elif key == ord('m'):
                for tracker in self.trackers:
                    tracker.has_mask = False

            elif key in [ord('t'), ord('.'), 85, 86]:
                # Start/stop a trial period
                if not self.ev_trial_active.is_set():
                    self.ev_trial_active.set()
                else:
                    self.ev_trial_active.clear()
                logging.info('Trial {}'.format(
                    '++++++++ active ++++++++' if self.ev_trial_active.is_set() else '------- inactive -------'))

            elapsed = ((cv2.getTickCount() - t0) / cv2.getTickFrequency()) * 1000
            self._loop_times.appendleft(elapsed)
            t0 = cv2.getTickCount()

            if not (frame_idx % 100):
                logging.debug(
                    'Display/tracking at {:.1f} fps'.format(1000 / (sum(self._loop_times) / len(self._loop_times))))

    def add_overlay(self, frame, t):
        """Overlay of time passed in normal/recording mode with recording indicator"""
        t_str = fmt_time(t)
        ox, oy = 4, 4
        # osx = 15
        thickness = 1
        font_scale = 1.2

        ts, bl = cv2.getTextSize(t_str, FONT, font_scale, thickness + 2)

        if not self.ev_recording.is_set():
            bg, fg = (0, 0, 0), (255, 255, 255)
            radius = 0
        else:
            bg, fg = (255, 255, 255), (0, 0, 0)
            radius = 8

        cv2.rectangle(frame, (ox - thickness, frame.shape[0] - oy + thickness),
                      (ox + ts[0] + 2 * radius, frame.shape[0] - oy - ts[1] - thickness), bg, cv2.FILLED)

        cv2.putText(frame, t_str, (ox, frame.shape[0] - oy), FONT, font_scale, fg,
                    thickness, lineType=cv2.LINE_AA)

        if self.ev_recording.is_set():
            cv2.circle(frame, (ox + ts[0] + radius, frame.shape[0] - ts[1] // 2 - oy), radius, (0, 0, 255), -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sources', nargs='*', help='List of sources to read from', default=FRAME_SOURCES)
    parser.add_argument('-d', '--debug', action='store_true', help='Debug mode')

    cli_args = parser.parse_args()

    logfile = Path.home() / "Videos/hextrack/{}_hextrack_log".format(
        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())))

    if cli_args.debug:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - (%(threadName)-9s) %(message)s')

    else:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - (%(threadName)-9s) %(message)s')

    fh = logging.FileHandler(str(logfile))
    fhf = logging.Formatter('%(asctime)s : %(levelname)s : [%(threadName)-9s] - %(message)s')
    fh.setFormatter(fhf)
    logging.getLogger('').addHandler(fh)

    # Construct the shared array to fit all frames
    width = FRAME_WIDTH
    height = FRAME_HEIGHT + FRAME_METADATA
    colors = FRAME_COLORS
    fps = FRAME_FPS
    n_views = len(cli_args.sources)
    num_bytes = width * height * colors * n_views

    SHARED_ARR = mp.Array(ctypes.c_ubyte, num_bytes)

    logging.debug('Created shared array: {}'.format(SHARED_ARR))
    logging.debug('Intended shared array shape {}x{}x{}'.format(width, height * n_views, colors))

    mp.freeze_support()

    ht = HexTrack(sources=cli_args.sources, shared_arr=SHARED_ARR, frame_shape=(width, height, colors))
    ht.loop()
