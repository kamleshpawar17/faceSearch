from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from face_detection import detectFace


class getFaceEmbeddings:
    def __init__(self, config: Dict) -> None:
        """class to extract embedding for the face

        Args:
            config (dict): dictionary of the config parameters
        """
        self.imgsz_fr = config["FE_IMAGE_SIZE"]
        self.net = cv2.dnn.readNetFromTorch(config["FE_MODEL"])
        self.net.setPreferableTarget(config["TARGET_COMPUTE"])
        self.min_face_size = config["MIN_FACE_SIZE"]
        self.scale_factor = config["FE_SCALE_FACTOR"]
        self.image_types = config["IMAGE_TYPES"]
        self.face_detector = detectFace(config)

    def get_128d_face_embeddings(
        self, frame: cv2.Mat, face_rects: List[np.array]
    ) -> List[np.array]:
        """function to compute face embedding from the input image with multiple faces and bounding boxes

        Args:
            frame (cv2.Mat): input image
            face_rects (List[np.array]): list of face bounding boxes within the input image

        Returns:
            List[np.array]: list of 128 dimensional face embeddings for each face
        """
        face_embeddings = []
        for face_rect in face_rects:
            # crop face image
            face_image = frame[face_rect[1] : face_rect[3], face_rect[0] : face_rect[2]]
            face_h, face_w, _ = face_image.shape
            # ignore small faces
            if (face_h < self.min_face_size) or (face_w < self.min_face_size):
                logger.warning(
                    f"size of face ({face_h}, {face_w}) is smaller than {self.min_face_size}, ignoring this instance"
                )
                continue
            # pass the cropped face thorough the network to compute face embeddings
            blob = cv2.dnn.blobFromImage(
                face_image,
                1.0 / 255,
                (self.imgsz_fr, self.imgsz_fr),
                (0, 0, 0),
                swapRB=True,
                crop=False,
            )
            self.net.setInput(blob)
            vec = self.scale_factor * self.net.forward()  # [1, 128]
            face_embeddings.append(vec)
        return face_embeddings

    def get_face_embeddings_from_dir(self, input_dir: str) -> Tuple[np.array, Dict]:
        """function to compute face embedding for all the images in the input directory

        Args:
            input_dir (str): path to input directory

        Returns:
            Tuple[np.array, List]: face embedding array for all the images, list of the names of the images
        """
        face_embeddings = []
        face_image_names, face_index = {}, 0
        input_dir = Path(input_dir)
        # loop through all types of image file
        number_of_images = 0
        for image_type in self.image_types:
            # loop through all images of one type
            fnames = list(input_dir.glob(image_type))
            number_of_images += len(fnames)
            for fname in tqdm(fnames):
                fname = str(fname)
                # read image
                frame = cv2.imread(fname)
                # detect faces
                face_rects = self.face_detector.detect_face(frame)
                # compute face embeddings
                if len(face_rects) > 0:
                    face_embedding = self.get_128d_face_embeddings(frame, face_rects)
                    face_embeddings.extend(face_embedding)
                    for k in range(face_index, face_index + len(face_embedding)):
                        face_image_names[str(k)] = fname.split("/")[-1]
                    face_index += len(face_embedding)
        face_embeddings = np.squeeze(np.array(face_embeddings))
        logger.info(f"Total number of input images were: {number_of_images}")
        return face_embeddings, face_image_names
