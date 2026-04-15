#!/usr/bin/env python3

"""
Object Classification with NNStreamer + TFLite
---------------------------------------------

This script runs Object Classification using GStreamer pipelines
with NNStreamer tensor_filter (TFLite backend). It supports CPU and NPU backends
on i.MX platforms and can render results with Wayland/X11/cairooverlay.

Author: <Your Name/Org>
License: MIT
"""

import os
import sys
import logging
import gi
import cairo
import argparse
import signal
import pathlib
import time
import subprocess
import re
import glob

gi.require_version('Gst', '1.0')
gi.require_foreign('cairo')
from gi.repository import Gst, GObject, GLib


class Object_Classification:
    """The class that manages the demo"""
    def __init__(self, platform, device, backend, save_video,
        model, labels, display, width, height,c_format,
        callback=None, r=1, g=0, b=0):
    
        """Creates an instance of the demo

        Arguments:
        device -- What camera or video file to use
        backend -- Whether to use NPU or CPU
        model -- the path to the model
        labels -- the path to the labels
        display -- Whether to use X11 or Weston
        callback -- Callback to pass stats to
        width -- Width of output
        height -- Height of output
        r -- Red value for labels
        g -- Green value for labels
        b -- Blue value for labels
        """
        self.loop = None
        self.pipeline = None
        self.running = False
        self.current_label_index = -1
        self.new_label_index = -1
        self.tflite_model = model
        self.label_path = labels
        self.device = device
        self.backend = backend
        self.display = display
        self.video = save_video
        self.c_format = c_format
        self.callback = callback
        self.tflite_labels = []
        self.VIDEO_WIDTH = width
        self.VIDEO_HEIGHT = height
        self.label = "Loading..."
        self.first_frame = True
        self.refresh_time = -1
        self.r = r
        self.b = b
        self.g = g
        self.platform = platform
        if not self.tflite_init():
            raise Exception
        GObject.threads_init()
        Gst.init(None)

    def _select_bitrate(self):
        px = self.VIDEO_WIDTH * self.VIDEO_HEIGHT
        if px >= 1920 * 1080:
            return 8000  # 8 Mbps
        elif px >= 1280 * 720:
            return 5000  # 5 Mbps
        else:
            return 2500  # SD

    def run_code(self):
        """Starts pipeline and run demo"""

        if self.backend == "CPU":
            backend = "true:cpu custom=NumThreads:4"
        elif self.backend == "GPU":
            os.environ["USE_GPU_INFERENCE"] = "1"
            backend = ("true:gpu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")
        else:
            os.environ["USE_GPU_INFERENCE"] = "0"
            backend = ("true:npu custom=Delegate:External,"
                       "ExtDelegateLib:libvx_delegate.so")
        save_video = ""               
        if self.video == "yes":
            save_video = (
                " tt. ! queue max-size-buffers=16 leaky=no "
                " ! videoconvert "
                " ! videorate "
                " ! video/x-raw,format=NV12,framerate=30/1 "
                " ! queue max-size-buffers=16 leaky=no "
                f" ! vpuenc_h264 bitrate={self._select_bitrate()} gop-size=30 "
                " ! h264parse "
                " ! mp4mux faststart=true "
                " ! filesink location=output_video.mp4 sync=false "
            )
            
        if self.display == "X11":
            display = "ximagesink name=img_tensor"
        elif self.display == "None":
            self.print_time = GLib.get_monotonic_time()
            display = "fakesink "
        else:
            display = "waylandsink sync=true name=img_tensor"

        self.past_time = GLib.get_monotonic_time()
        self.interval_time = -1
        self.label_time = GLib.get_monotonic_time()

        if self.platform == "imx8qmmek":
            decoder = "h264parse ! v4l2h264dec ! imxvideoconvert_g2d "
        else:
            decoder = "vpudec "
        
        if self.device.startswith("rtsp://"):
            pipeline = 'rtspsrc location=' + self.device
            pipeline += ' ! rtph264depay ! vpudec'
            pipeline += ' ! videoconvert ! tee name=t_raw '
        elif "/dev/video" in self.device and self.c_format == 'YUY2':
            pipeline = 'v4l2src name=cam_src device=' + self.device
            pipeline += ' ! videoconvert ! video/x-raw,width='
            pipeline += str(int(self.VIDEO_WIDTH)) +',height='
            pipeline += str(int(self.VIDEO_HEIGHT))
            pipeline += ',format='
            pipeline += self.c_format
            pipeline += '! tee name=t_raw'
        elif "/dev/video" in self.device:
            pipeline = 'v4l2src name=cam_src device=' + self.device
            pipeline += ' ! imxvideoconvert_g2d ! video/x-raw,width='
            pipeline += str(int(self.VIDEO_WIDTH)) +',height='
            pipeline += str(int(self.VIDEO_HEIGHT))
            pipeline += ',format='
            pipeline += self.c_format
            pipeline += '! tee name=t_raw'
        else:
            pipeline = 'filesrc location=' + self.device  + ' ! qtdemux'
            pipeline += ' ! ' + decoder + '! tee name=t_raw'
        # main loop
        self.loop = GObject.MainLoop()
        pipeline += ' t_raw. ! imxvideoconvert_g2d ! '
        pipeline += 'video/x-raw,width=224,height=224,format=RGBA ! '
        pipeline += 'queue max-size-buffers=2 leaky=2 ! '
        pipeline += 'videoconvert ! video/x-raw,format=RGB ! '
        pipeline += 'tensor_converter ! tensor_filter name=tensor_filter '
        pipeline += 'framework=tensorflow-lite model='
        pipeline += self.tflite_model + ' accelerator=' + backend
        pipeline += ' silent=FALSE latency=1 ! tensor_sink name=tensor_sink'
        pipeline += ' t_raw. ! imxvideoconvert_g2d ! cairooverlay '
        pipeline += 'name=tensor_res draw-on-transparent-surface=false ! '
        pipeline += 'tee name=tt tt. ! queue max-size-buffers=2 leaky=2 ! ' + display + save_video

        # init pipeline
        self.pipeline = Gst.parse_launch(pipeline)
        print(pipeline)
        # bus and message callback
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_bus_message)

        self.tensor_filter = self.pipeline.get_by_name('tensor_filter')

        # tensor sink signal : new data callback
        tensor_sink = self.pipeline.get_by_name('tensor_sink')
        tensor_sink.connect('new-data', self.on_new_data)

        self.reload_time = GLib.get_monotonic_time()
        tensor_res = self.pipeline.get_by_name('tensor_res')
        tensor_res.connect('draw', self.draw_overlay_cb)
        tensor_res.connect('caps-changed', self.prepare_overlay_cb)

        # start pipeline
        self.pipeline.set_state(Gst.State.PLAYING)
        self.running = True

        if self.callback is not None:
            GObject.timeout_add(500, self.callback, self)

        # set window title
        self.set_window_title('img_tensor', 'Object Classification')

        def handle_sigint(sig, frame):
            print("\nCtrl+C detected → Sending EOS to finalize MP4...")
            self.pipeline.send_event(Gst.Event.new_eos())

            msg = bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS)
            if msg:
                print("EOS received → pipeline ended cleanly")
            else:
                print("Timeout waiting for EOS → forcing exit")

            self.loop.quit()

        signal.signal(signal.SIGINT, handle_sigint)

        # ---- Run main loop ----
        try:
            self.loop.run()
        finally:
            self.running = False
            self.pipeline.set_state(Gst.State.NULL)
            bus.remove_signal_watch()
            sys.exit(0)

    def on_bus_message(self, bus, message):
        """Callback for message.

        :param bus: pipeline bus
        :param message: message from pipeline
        :return: None
        """
        if message.type == Gst.MessageType.EOS:
            logging.info('received eos message')
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            error, debug = message.parse_error()
            logging.warning('[error] %s : %s', error.message, debug)
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            error, debug = message.parse_warning()
            logging.warning('[warning] %s : %s', error.message, debug)
        elif message.type == Gst.MessageType.STREAM_START:
            logging.info('received start message')
        elif message.type == Gst.MessageType.QOS:
            data_format, processed, dropped = message.parse_qos_stats()
            format_str = Gst.Format.get_name(data_format)
            logging.debug('[qos] format[%s] processed[%d] dropped[%d]',
                format_str, processed, dropped)

    def on_new_data(self, sink, buffer):
        """Callback for tensor sink signal.

        :param sink: tensor sink element
        :param buffer: buffer from element
        :return: None
        """
        if self.running:
            new_time = GLib.get_monotonic_time()
            self.interval_time = new_time - self.past_time
            self.past_time = new_time

            for idx in range(buffer.n_memory()):
                mem = buffer.peek_memory(idx)
                result, mapinfo = mem.map(Gst.MapFlags.READ)
                if result:
                    # update label index with max score
                    self.update_top_label_index(mapinfo.data, mapinfo.size)
                    mem.unmap(mapinfo)
            if self.current_label_index != self.new_label_index:
                # update textoverlay
                self.current_label_index = self.new_label_index
                if(GLib.get_monotonic_time() - self.label_time > 10000):
                    self.label = self.tflite_get_label(
                        self.current_label_index)[:-1]
                    self.label_time = GLib.get_monotonic_time()

            if self.display == "None":
                if (GLib.get_monotonic_time() - self.print_time) > 1000000:
                    inference = self.tensor_filter.get_property("latency")
                    print(
                        "Item: " + self.label + " Inference time: " +
                        str(inference/1000) + " ms (" +
                        "{:5.2f}".format(1/(inference/1000000)) + " IPS)")
                    self.print_time = GLib.get_monotonic_time()


    def set_window_title(self, name, title):
        """Set window title.

        :param name: GstXImageSink element name
        :param title: window title
        :return: None
        """
        element = self.pipeline.get_by_name(name)
        if element is not None:
            pad = element.get_static_pad('sink')
            if pad is not None:
                tags = Gst.TagList.new_empty()
                tags.add_value(Gst.TagMergeMode.APPEND, 'title', title)
                pad.send_event(Gst.Event.new_tag(tags))


    # Modified: Changed filepath to point to model and lables on board.
    def tflite_init(self):
        """Check tflite model and load labels.

        :return: True if successfully initialized
        """

        # check model file exists
        if not os.path.exists(self.tflite_model):
            logging.error('cannot find tflite model [%s]', self.tflite_model)
            return False

        # load labels
        label_path = self.label_path
        try:
            with open(label_path, 'r') as label_file:
                for line in label_file.readlines():
                    self.tflite_labels.append(line)
        except FileNotFoundError:
            logging.error('cannot find tflite label [%s]', label_path)
            return False

        logging.info(
            'finished to load labels, total [%d]', len(self.tflite_labels))
        return True

    def tflite_get_label(self, index):
        """Get label string with given index.

        :param index: index for label
        :return: label string
        """
        try:
            label = self.tflite_labels[index]
        except IndexError:
            label = ''
        return label

    def update_top_label_index(self, data, data_size):
        """Update tflite label index with max score.

        :param data: array of scores
        :param data_size: data size
        :return: None
        """
        # -1 if failed to get max score index
        self.new_label_index = -1
        if data_size == len(self.tflite_labels):
            scores = [data[i] for i in range(data_size)]
            max_score = max(scores)
            if max_score > 0:
                self.new_label_index = scores.index(max_score)
        else:
            logging.error('unexpected data size [%d]', data_size)

    def draw_overlay_cb(self, overlay, context, timestamp, duration):
        """Draws the results onto the video frame"""
        scale_height = self.VIDEO_HEIGHT/1080
        scale_width = self.VIDEO_WIDTH/1920
    
        scale_text = max(scale_height, scale_width)
        inference = self.tensor_filter.get_property("latency")
        context.select_font_face(
            'Sans', cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        context.set_source_rgb(0, 0, 1)
        context.move_to(int(50 * scale_width), int(100 * scale_height))
        context.set_font_size(int(100 * scale_text))
        context.show_text(self.label)
        if(self.first_frame):
            context.set_source_rgb(1,1,1)
            context.move_to(int(400 * scale_width), int(600 * scale_height))
            context.set_font_size(int(200.0 * min(scale_width,scale_height)))
            self.first_frame = False
        context.set_operator(cairo.Operator.SOURCE)

    def prepare_overlay_cb(self, overlay, caps):
        self.video_caps = caps
ALLOWED_RESOLUTIONS = [(640, 360),(640, 480), (1280, 720), (1920, 1080)]

def get_video_nodes():
    return glob.glob("/dev/video*")

def is_streaming(dev, num_buffers=3):
    """Check if the video node can stream."""
    try:
        subprocess.run(
            ["timeout", "5", "gst-launch-1.0", "-q",
             "v4l2src", f"device={dev}", f"num-buffers={num_buffers}", "!", "fakesink"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_format_info(dev):
    """Return only supported (w,h,pixfmt) among 640x480, 1280x720, 1920x1080."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", dev, "--list-formats-ext"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        formats = []
        current_pixfmt = None

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("["):
                pixfmt_match = re.search(r"'(\w+)'", line)
                if pixfmt_match:
                    current_pixfmt = pixfmt_match.group(1)
            elif "Size:" in line and "Discrete" in line:
                size_match = re.search(r"(\d+)x(\d+)", line)
                if size_match and current_pixfmt:
                    w, h = map(int, size_match.groups())
                    if (w, h) in ALLOWED_RESOLUTIONS and current_pixfmt in ("YUYV", "BGRx", "MJPG"):
                        formats.append((w, h, current_pixfmt))
        return formats
    except subprocess.CalledProcessError:
        return []

def select_webcam():
    print("Available supported video nodes for streaming:")
    print("-------------------------------------")
    nodes = get_video_nodes()
    working_nodes = []

    # Check each node
    for dev in nodes:
        if is_streaming(dev):
            formats = get_format_info(dev)
            if not formats:
                continue
            print(f"{len(working_nodes)+1}. {dev}")
            for idx, (w, h, fmt) in enumerate(formats, start=1):
                print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")
            working_nodes.append((dev, formats))

    if not working_nodes:
        print("No working video nodes found with standard resolutions.")
        return

    # Ask user to select node
    while True:
        choice = input(f"\nSelect a video node [1-{len(working_nodes)}]: ")
        if choice.isdigit():
            choice = int(choice)
            if 1 <= choice <= len(working_nodes):
                selected_node, formats = working_nodes[choice-1]
                break
        print("Invalid choice, try again.")

    # Show available standard resolutions for selected node
    print(f"\nAvailable supported resolutions for {selected_node}:")
    for idx, (w, h, fmt) in enumerate(formats, start=1):
        print(f"   [{idx}] {w}x{h}, PixelFormat: {fmt}")

    # Ask user to select resolution
    while True:
        res_choice = input(f"Select a resolution [1-{len(formats)}]: ")
        if res_choice.isdigit():
            res_choice = int(res_choice)
            if 1 <= res_choice <= len(formats):
                width, height, pixfmt = formats[res_choice-1]
                break
        print("Invalid choice, try again.")

    # Display final selection
    print("\nSelected video node parameters:")
    print(f"Device: {selected_node}")
    print(f"Resolution: {width}x{height}")
    print(f"Pixel Format: {pixfmt}")
    print(f"Framerate: 30/1")

    return {
        "device": selected_node,
        "width": width,
        "height": height,
        "pixfmt": pixfmt,
        "framerate": 30
    }

            
def get_resolution(filepath: str):
    """
    Get resolution (width, height) of a video file or RTSP stream
    using gst-discoverer-1.0.
    Returns (width, height) or (None, None) on failure.
    """
    try:
        result = subprocess.run(
            ["gst-discoverer-1.0", filepath],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        width, height = None, None
        for line in result.stdout.splitlines():
            line = line.strip()
            # Match "Width: 1920", "Height: 1088", etc.
            m_w = re.search(r"Width\s*:\s*(\d+)", line, re.IGNORECASE)
            m_h = re.search(r"Height\s*:\s*(\d+)", line, re.IGNORECASE)
            if m_w:
                width = int(m_w.group(1))
            if m_h:
                height = int(m_h.group(1))

        if width is None or height is None:
            print(f"[WARNING] Could not detect resolution for {filepath}")
            return None, None

        print(f"[INFO] Detected resolution: {width}x{height}")
        return width, height

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] gst-discoverer-1.0 failed: {e}")
        return None, None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Object Classification (NNStreamer + TFLite)")
    parser.add_argument("--video", default="/workspace/videos/id_example.mov",help="Video source: 'discover', RTSP URL (rtsp://<ip>:<port>/<path>), or video file path")
    parser.add_argument("--platform", default="imx8mp", help="Target platform")
    parser.add_argument("--backend", choices=["CPU", "NPU"], default="CPU", help="Inference backend")
    parser.add_argument("--model", default="/workspace/models/mobilenet_v1_1.0_224_quant.tflite", help="TFLite model path")
    parser.add_argument("--labels", default="/workspace/models/imagenet_labels.txt", help="Labels file")
    parser.add_argument("--save-video", choices=["yes", "no"], default="no", help="Save output video (default: no)")
    parser.add_argument("--display", choices=["X11", "weston"], default="weston", help="Display backend (default: weston)")
    parser.add_argument("--cam-width", type=int, default=1920,help="Camera input width (default: 1920)")
    parser.add_argument("--cam-height", type=int, default=1080,help="Camera input height (default: 1080)")
    parser.add_argument("--cam-format", default="BGRx",help="Camera pixel format (default: BGRx)")
    args = parser.parse_args()
    if args.display == "X11":
        args.save_video = "no"

    # Video source handling
    if args.video.lower() == "discover":
        cam_params = select_webcam()
        args.video = cam_params["device"]
        w, h = cam_params["width"], cam_params["height"]
        args.cam_format = cam_params["pixfmt"]
        if args.cam_format == "YUYV":
            args.cam_format = "YUY2"
        else:
            args.cam_format = "BGRx"
    elif args.video.startswith("rtsp://") or not args.video.startswith("/dev/video"):
        # RTSP stream or video file: probe resolution using gst-discoverer-1.0
        w, h = get_resolution(args.video)
        if w is None or h is None:
            # fallback if probe fails
            w, h = args.cam_width, args.cam_height
            print(f"[WARNING] Could not detect resolution, using defaults {w}x{h}")
        print(f"Stream/Video selected -> {args.video}, resolution: {w}x{h}")
    
    elif args.video.startswith("/dev/video"):
        # Direct camera device: use provided width/height
        w, h = args.cam_width, args.cam_height
        print(f"Direct camera device detected -> {args.video}, resolution {w}x{h}")

    if not w or not h or w <= 0 or h <= 0:
        print(f"[ERROR] Invalid resolution detected: width={w}, height={h}")
        sys.exit(1)

    # Print configuration
    print("========== Configuration ==========")
    print(f"Video Source : {args.video}")
    print(f"Platform     : {args.platform}")
    print(f"Backend      : {args.backend}")
    print(f"Model File   : {args.model}")
    print(f"Label File   : {args.labels}")
    print(f"Save Video   : {args.save_video}")
    print(f"Display      : {args.display}")
    print(f"Width        : {w}")
    print(f"Height       : {h}")
    print(f"Format       : {args.cam_format}")
    print("===================================")
    
    est = Object_Classification(
        platform=args.platform,
        device=args.video,
        backend=args.backend,
        model=args.model,
        labels=args.labels,
        save_video=args.save_video,
        display=args.display,
        width=w,
        height=h,
        c_format=args.cam_format
    )
    est.run_code()
