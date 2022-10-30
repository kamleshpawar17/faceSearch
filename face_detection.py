from typing import List

import cv2
import numpy as np


class detectFace:
    def __init__(self, config: dict) -> None:
        """face detection class initialization and methods

        Args:
            config (dict): dictionary of the config parameters
        """
        self.imgsz_fd = config["FD_IMAGE_SIZE"]
        self.net = cv2.dnn.readNetFromCaffe(config["FD_PROTO"], config["FD_WEIGHTS"])
        self.net.setPreferableTarget(config["TARGET_COMPUTE"])
        self.threshold = config["FD_THRESHOLD"]
        self.fd_box_scale = config["FD_BOX_SCALE"]

    def detect_face(self, frame: cv2.Mat) -> List[np.array]:
        """face detection function

        Args:
            frame (cv2.Mat): input image matrix

        Returns:
            List: list of bounding boxes for all detected faces [start_x, start_y, end_x, end_y]
        """
        H, W, _ = frame.shape
        # convert the cv2 image matrix to blob for the fd network and resize + scale intensities
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (self.imgsz_fd, self.imgsz_fd)),
            1.0,
            (self.imgsz_fd, self.imgsz_fd),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        # forward pass
        self.net.setInput(blob)
        # numpy shape [1, 1, 200, 7] for first 200 bb
        detections = self.net.forward()
        # number of positive faces
        indx_face = detections[0, 0, :, 2] > self.threshold
        number_of_faces = np.sum(indx_face)
        # if there are faces loop through them and find bounding boxes
        face_rects = []
        if number_of_faces:
            detections = detections[:, :, indx_face, :]
            for k in range(number_of_faces):
                # box: [start_x, start_y, end_x, end_y]
                # self.fd_box_scale modifies the box as
                # (detections[0, 0, k, 3:] + (self.fd_box_scale * np.array([-1.0, -1.0, 1.0, 1.0])) *  detections[0, 0, k, 3:])
                box = (
                    detections[0, 0, k, 3:]
                    + (self.fd_box_scale * np.array([-1.0, -1.0, 1.0, 1.0]))
                    * detections[0, 0, k, 3:]
                ) * np.array([W, H, W, H])
                # clip out of bound bounding boxes
                box[[0, 2]] = np.clip(box[[0, 2]], a_min=0, a_max=W - 1)
                box[[1, 3]] = np.clip(box[[1, 3]], a_min=0, a_max=H - 1)
                face_rects.append(box.astype("int"))
        return face_rects
