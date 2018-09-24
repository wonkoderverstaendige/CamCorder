import ctypes
import cv2
import threading
from queue import Queue, Empty, Full
import time
import logging
from pathlib import Path
import multiprocessing as mp
import numpy as np

from camcorder.lib.framesources import Frame

logging.basicConfig(level=logging.DEBUG, format='(%(threadName)-9s) %(message)s', )

N_FRAMES_LOG_WINDOW = 100
N_FRAMES_FPS_WINDOW = 10

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_N_BYTES = FRAME_HEIGHT * FRAME_WIDTH * 3
FRAME_FPS = 15.

# shared buffer for transferring frames between threads/processes
shared_arr = mp.Array(ctypes.c_ubyte, FRAME_N_BYTES)
logging.debug('Created shared array: {}'.format(shared_arr))


def buf_to_numpy(buf, shape):
    """Return numpy object from a raw buffer, e.g. multiprocessing Array"""
    return np.frombuffer(buf.get_obj(), dtype=np.ubyte).reshape(shape)


class Grabber(threading.Thread):
    def __init__(self, out_queue, trigger_event):  # , in_queue, out_queue
        super().__init__()
        self.name = 'Grabber'

        self.n_frames = 0
        self.capture = None
        self.frame = None

        self.width = FRAME_WIDTH
        self.height = FRAME_HEIGHT

        global shared_arr
        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(shared_arr))
            self._fresh_frame = buf_to_numpy(shared_arr, (FRAME_HEIGHT, FRAME_WIDTH, -1))
            logging.debug('Numpy shared buffer at {}'.format(hex(self._fresh_frame.ctypes.data)))

        self._write_queue = out_queue
        self._trigger = trigger_event
        self._avg_fps = FRAME_FPS

        logging.debug('Grabber initialization done!')

    def run(self):
        logging.debug('Starting loop in {}!'.format(self.name))
        self.capture = cv2.VideoCapture(0)

        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # MAXIMUM fps. This is more a recommendation, most cameras don't listen to this.
        self.capture.set(cv2.CAP_PROP_FPS, FRAME_FPS)

        t0 = cv2.getTickCount()

        while not self._trigger.is_set():
            rt, frame = self.capture.read()
            self.frame = Frame(self.n_frames, frame, 'Grabber', add_timestamp=True, add_tickstamp=True)

            elapsed = (cv2.getTickCount() - t0) / cv2.getTickFrequency() * 1000
            t0 = cv2.getTickCount()
            self._avg_fps = (1 - 1 / N_FRAMES_FPS_WINDOW) * self._avg_fps + (1 / N_FRAMES_FPS_WINDOW) * 1000 / elapsed

            # Every now and then show fps
            if not self.n_frames % N_FRAMES_LOG_WINDOW:
                logging.debug(
                    'Grabbing frame {}... {} after {:.2f} ms, {:.1f} fps'.format(self.n_frames, 'OK' if rt else 'FAIL',
                                                                                 elapsed, self._avg_fps))

            # Send frames to attached threads/processes
            self.relay_frames()

            self.n_frames += 1

        logging.debug('Stopping loop in {}!'.format(self.name))

    def relay_frames(self):
        # Forward frame to Writer via Queue
        try:
            self._write_queue.put(self.frame, timeout=.5)
        except Full:
            logging.warning('Dropped frame {}'.format(self.frame.index))

        # Forward frame for tracking and display
        # NOTE: [:] indicates to reuse the buffer
        with self._shared_arr.get_lock():
            self._fresh_frame[:] = self.frame.img


class Writer(threading.Thread):
    def __init__(self, in_queue, ev_alive, ev_recording):
        super().__init__()
        self.name = 'Writer'
        self.writer = None

        self.n_frames = 0
        self.width = None
        self.height = None
        self.frame = None

        self.in_queue = in_queue
        self._ev_alive = ev_alive
        self._ev_recording = ev_recording

        self.recording = False

        self.codec = 0x00000021  # cv2.VideoWriter_fourcc(*'MP4V')
        self.container = '.mp4'
        self.video_fname = str(Path("C:/Users/reichler/Videos/camcorder/testing_{}" + self.container))

        logging.debug('Writer initialization done!')

    def run(self):
        logging.debug('Starting loop in {}!'.format(self.name))
        while not self._ev_alive.is_set():
            try:
                self.frame = self.in_queue.get(timeout=.5)
            except Empty:
                continue
            # logging.debug('Got frame {}!'.format(self.frame.index))
            h, w = self.frame.img.shape[:2]
            if self.recording != self._ev_recording.is_set():
                self.recording = self._ev_recording.is_set()
                if self.recording:
                    ts = time.strftime("%d-%b-%y_%H-%M-%S", time.localtime(time.time()))
                    vname = self.video_fname.format(ts)

                    logging.debug('Starting Recording to {}'.format(vname))
                    self.writer = cv2.VideoWriter(vname, self.codec, 15., (w, h))
                else:
                    logging.debug('Stopping Recording')
                    self.writer.release()

            if self.recording:
                self.writer.write(self.frame.img)

        logging.debug('Stopping loop in {}!'.format(self.name))
        if self.writer is not None:
            self.writer.release()


class Tracker:
    def __init__(self):
        super().__init__()

    def track(self, frame):
        pass


class HexTrack:
    def __init__(self):
        threading.current_thread().name = 'HexTrack'

        self.write_queue = Queue(maxsize=16)
        self.ev_kill_grabber = threading.Event()
        self.ev_kill_writer = threading.Event()
        self.ev_recording = threading.Event()
        self.ev_tracking = threading.Event()

        # Frame acquisition object
        self.grabber = Grabber(out_queue=self.write_queue,
                               trigger_event=self.ev_kill_grabber)
        self.grabber.start()
        self.frame = None
        global shared_arr
        with shared_arr.get_lock():
            self._shared_arr = shared_arr
            logging.debug('Grabber shared array: {}'.format(self._shared_arr))

        # Video storage
        self.writer = Writer(in_queue=self.write_queue, ev_alive=self.ev_kill_writer, ev_recording=self.ev_recording)
        self.writer.start()

        # Online tracker
        self.tracker = Tracker()

        logging.debug('HexTrack initialization done!')

    def run(self):
        while self.grabber.is_alive():
            try:
                # self.frame = self.frame_queue.get(timeout=.5)
                with self._shared_arr.get_lock():
                    self.frame = buf_to_numpy(self._shared_arr, (FRAME_HEIGHT, FRAME_WIDTH, 3))
            except Empty:
                continue

            if self.frame is not None:
                cv2.imshow('frame', self.frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    self.ev_kill_grabber.set()
                    logging.debug('Grabber join request sent!')

                    self.ev_kill_writer.set()
                    logging.debug('Writer join request sent!')

                    self.grabber.join()
                    logging.debug('Grabber joined!')

                    self.writer.join()
                    logging.debug('Writer joined!')

                    break

                elif key == ord('r'):
                    if not self.ev_recording.is_set():
                        self.ev_recording.set()
                    else:
                        self.ev_recording.clear()


if __name__ == '__main__':
    mp.freeze_support()
    ht = HexTrack()
    ht.run()
