"""camera.py

This code is a tiny and custom version of https://github.com/jkjung-avt/tensorrt_demos#yolov4
"""

import cv2
import numpy as np
import threading


USB_GSTREAMER = True

def add_camera_args(parser):
    """Add parser augment for camera options."""

    parser.add_argument("--video_path", 
                            type = str,
                            default = None,
                            help = "use a video file as input")

    parser.add_argument("--image_path", 
                        type = str,
                        default = None,
                        help = "use an image file as input",
                        required  = False)
    parser.add_argument("--video_dev",
                        type = int,
                        default = None, 
                        help = "device number e.g.: 0",
                        required = False)
                    
    parser.add_argument("--width",
                        dest="image_width",
                        help = "image width value",
                        default = 640,
                        type = int)

    parser.add_argument("--height",
                    dest="image_height",
                    help = "image height value",
                    default = 480,
                    type = int)
    return parser


def open_cam_usb(dev, width, height):
    """Open a USB webcam"""
    if USB_GSTREAMER:
        gst_str = ("v4l2src device=/dev/video{} ! "
                   "video/x-raw, width=(int){}, height=(int){} ! "
                   "videoconvert ! appsink").format(dev, width, height)
        return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
    else:
        return cv2.VideoCapture(dev)


def open_cam_onboard(width, height):
    """Open the Jetson onboard camera."""
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, you might need to add
        # 'flip-method=2' into gst_str below.
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def grab_img(cam):
    """This 'grab_img' function is designed to be run in the sub-thread.
    Once started, this thread continues to grab a new image and put it
    into the global 'img_handle', until 'thread_running' is set to False.
    """
    while cam.thread_running:
        _, cam.img_handle = cam.cap.read()
        if cam.img_handle is None:
            logging.warning('grab_img(): cap.read() returns None...')
            break
    cam.thread_running = False

class Camera():
    """Camera class which supports reading images from theses video sources:
    1. Video file
    2. Image (jpg, png, etc.) file, repeating indefinitely
    3. USB webcam
    4. Jetson onboard camera
    """

    def __init__(self, args):
        self.args = args
        self.is_opened = False
        self.use_thread = False
        self.thread_running = False
        self.img_handle = None
        self.img_width = 0
        self.img_height = 0
        self.cap = None
        self.thread = None

    def open(self):
        """Open camera based on command line arguments."""
        assert self.cap is None, 'Camera is already opened!'
        args = self.args

        if args.video_path is not None:
            self.cap = cv2.VideoCapture(args.video_path)
            self.use_thread = False

        elif args.image_path is not None:
            self.cap = "OK"
            self.img_handle = cv2.imread(args.image_path)
            if self.img_handle is not None:
                self.is_opened = True
                self.img_height, self.img_width, _ = self.img_handle.shape
            self.use_thread = False

        elif args.video_dev is not None:
            self.cap = open_cam_usb(
                args.video_dev,
                args.image_width,
                args.image_height
            )
            self.use_thread = True
        else:  # by default, use the jetson onboard camera
            self.cap = open_cam_onboard(
                args.image_width,
                args.image_height
            )
            self.use_thread = True

        if self.cap != "OK":
            if self.cap.isOpened():
                _, img = self.cap.read()
                if img is not None:
                    self.img_height, self.img_width, _ = img.shape
                    self.is_opened = True

    def start(self):
        assert not self.thread_running
        if self.use_thread:
            self.thread_running = True
            self.thread = threading.Thread(target=grab_img, args=(self,))
            self.thread.start()

    def stop(self):
        self.thread_running = False
        if self.use_thread:
            self.thread.join()

    def read(self):
        if  self.args.video_path is not None:
            _, img = self.cap.read()
            if img is None:
                #logging.warning('grab_img(): cap.read() returns None...')
                # looping around
                self.cap.release()
                self.cap = cv2.VideoCapture(self.args.video_path)
                _, img = self.cap.read()
            return img

        elif self.args.image_path is not None:
            return np.copy(self.img_handle)
        else:
            return self.img_handle

    def release(self):
        assert not self.thread_running
        if self.cap != 'OK':
            self.cap.release()