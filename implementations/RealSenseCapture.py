# RealSense
import pyrealsense2 as rs
import numpy as np
import cv2

# pytorch
import torch

# my model
from models.DivExYOLO import DivExYOLOVGG16
from dataloader.Loader import LoadDivRGBDFromCamera
from utils import TransformOutput as TO

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: ", depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# DivExYOLO settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DivExYOLOVGG16()
model = model.to(device)
MODEL_PATH = \
    '../outputs/2020-03-12/DivExYOLO_2020-03-12_Epoch130.pth'
model.load_state_dict(torch.load(MODEL_PATH))
load_divrgbd_from_camera = LoadDivRGBDFromCamera(img_size=224,
                                                 div_num=14,
                                                 depth_max=10000,
                                                 device=device)
yolo_output_to_bb = TO.YOLOOutput2BB(grid_scale=14,
                                     x_scale=20,
                                     y_scale=5,
                                     z_scale=10,
                                     conf_thresh=0.5,
                                     device=device)
save_data = TO.SaveData(color_save_flag=True,
                        depth_save_flag=True,
                        txt_save_flag=True,
                        save_dir='/mnt/HDD1/mtakahashi/process_space')

# # intrinsics
# frames = pipeline.wait_for_frames()
# color_ = frames.get_color_frame()
# intrinsics = color_.profile.as_video_stream_profile().intrinsics


model.eval()
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()  # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # transform images to the style of model input
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        rgbd = load_divrgbd_from_camera(color_image, depth_image)
        with torch.no_grad():
            rgbd = rgbd.to(device)
            output = model(rgbd)
        # Transform output
        bb = yolo_output_to_bb(output)

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth', depth_colormap)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
        elif key & 0xFF == ord('s'):
            save_data(save_name='out', color=color_image, depth=depth_image, bb=bb)
finally:
    pipeline.stop()