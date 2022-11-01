import pickle
import os
from pathlib import Path
from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from face_detection import detectFace
from face_embedding import getFaceEmbeddings
import hnswlib


class faceSearch:
    """face search class for creating a face search model using face detection anf face embeddings.
    The model for knn search is scann library
    """

    def __init__(self, config: dict) -> None:
        self.hnswlib_model_path = config["HNSWLIB_MODEL"]
        self.image_types = config["IMAGE_TYPES"]
        self.face_image_names_path = config["FACE_IMAGE_NAMES"]
        self.space = config["SPACE"]
        self.m = config["M"]
        self.ef_construction = config["EF_CONSTRUCTION"]
        self.ef = config["EF"]
        self.dim = config["VEC_DIM"]
        self.overwrite_model = config["OVERWRITE_MODEL"]
        self.face_detector = detectFace(config)
        self.face_embedder = getFaceEmbeddings(config)
        if Path(self.hnswlib_model_path).exists():
            self.searcher = hnswlib.Index(space=self.space, dim=self.dim)
            self.searcher.load_index(self.hnswlib_model_path)
            with open(self.face_image_names_path, "rb") as file:
                self.face_image_names = pickle.load(file)

    def register_faces_from_dir(self, input_dir: str) -> None:
        """this function:
        1. runs face detection
        2. extract face embeddings from detected faces
        3. creates the hnswlib model using face embeddings
        4. optionally saves the model

        Args:
            input_dir (str): input directory with all the images of faces
        """
        # get face embeddings
        logger.info("calculating face embeddings for input images....")
        (
            self.face_embeddings,
            self.face_image_names,
        ) = self.face_embedder.get_face_embeddings_from_dir(input_dir)
        logger.info(
            f"completed calculating {len(self.face_embeddings)} face embeddings"
        )

        # create hnswlib model
        logger.info("creating hnswlib model....")
        # Declaring index
        self.searcher = hnswlib.Index(space=self.space, dim=self.dim)
        # init index
        self.searcher.init_index(
            max_elements=len(self.face_embeddings),
            ef_construction=self.ef_construction,
            M=self.m,
        )
        self.searcher.set_ef(self.ef)
        # put data for indexing
        self.searcher.add_items(
            self.face_embeddings, np.arange(len(self.face_embeddings))
        )
        if self.overwrite_model:
            logger.info(f"saving hnswlib model to {self.hnswlib_model_path}")
            self.searcher.save_index(self.hnswlib_model_path)
            with open(self.face_image_names_path, "wb") as file:
                pickle.dump(self.face_image_names, file)
        else:
            logger.warning(f"Not saving hnswlib model")

    def search_similar_faces(self, input_image: str, number_of_images: int) -> List:
        """function to search similar faces for the given face

        Args:
            input_image (str): input image face
            number_of_images (int): number of similar faces to find

        Returns:
            List: list of the names of images consisting of the closely matching faces to the input image
        """
        logger.info(f"running face search query on image: {input_image}")
        # read image
        frame = cv2.imread(input_image)
        # detect faces
        face_rects = self.face_detector.detect_face(frame)
        logger.info(f"number of faces detected: {len(face_rects)}")
        # compute face embeddings
        if len(face_rects) > 0:
            face_embedding = np.squeeze(
                np.array(self.face_embedder.get_128d_face_embeddings(frame, face_rects))
            )
            if len(face_embedding.shape) == 1:
                face_embedding = np.expand_dims(face_embedding, 0)

            # knn search using hnswlib model
            neighbors, distances = self.searcher.knn_query(
                face_embedding, k=number_of_images
            )
            neighbors, distances = np.squeeze(neighbors), np.squeeze(distances)
            face_names_index = neighbors.astype("int").astype("str")
            similar_faces_names = list(map(self.face_image_names.get, face_names_index))

            for k in zip(neighbors, similar_faces_names, distances):
                logger.debug(f"index, name, distance : {k}")

            return similar_faces_names

    def show_images(
        self, ref_image: str, similar_faces_names: List[str], base_path: str
    ):
        """function to display the input and matched images

        Args:
            ref_image (str): path tot he input image for searching
            similar_faces_names (List[str]): names of the similar/matched face images
            base_path (str): base path to the image directory
        """
        number_of_images = len(similar_faces_names)
        cols = 5
        rows = np.ceil(number_of_images / cols) + 1
        frame = cv2.imread(ref_image)
        plt.figure(figsize=(7, 8))
        plt.subplot(rows, cols, 3)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("input image")
        plt.axis("off")

        for k in range(number_of_images):
            plt.subplot(rows, cols, k + cols + 1)
            frame = cv2.imread(os.path.join(base_path, similar_faces_names[k]))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")

        plt.tight_layout()
        plt.show()
