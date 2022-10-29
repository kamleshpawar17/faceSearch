import pickle
import os
from pathlib import Path
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from face_detection import detectFace
from face_embedding import getFaceEmbeddings
import scann


class faceSearch:
    """face search class for creating a face serach model using face detection anf face embeddings.
    The model for knn serach is scann library
    """

    def __init__(self, config: dict) -> None:
        self.model_path = config["SCANN_MODEL"]
        self.image_types = config["IMAGE_TYPES"]
        self.num_neighbours = config["NUMBER_OF_NEIGHBOURS"]
        self.num_leaves = config["NUMBER_OF_LEAVES"]
        self.face_image_names_path = config["FACE_IMAGE_NAMES"]
        self.num_leaves_to_search = config["NUMBER_OF_LEAVES_TO_SEARCH"]
        self.anisotropic_quatization_thrshld = config[
            "ANISOTROPIC_QUANTIZATION_THRSHLD"
        ]
        self.overwrite_model = config["OVERWRITE_MODEL"]
        self.face_detector = detectFace(config)
        self.face_embedder = getFaceEmbeddings(config)
        if Path(self.model_path).exists():
            self.searcher = scann.scann_ops_pybind.load_searcher(self.model_path)
            with open(self.face_image_names_path, "rb") as file:
                self.face_image_names = pickle.load(file)
        else:
            os.makedirs(self.model_path, exist_ok=True)

    def register_faces_from_dir(self, input_dir: str) -> None:
        """this function:
        1. runs face detection
        2. extract face embeddings from detected faces
        3. creates the scann model using face embeddings
        4. optionally saves the model

        Args:
            input_dir (str): input directoty with all the images of faces
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

        # create scann model
        logger.info("creating scann model....")
        self.searcher = (
            scann.scann_ops_pybind.builder(
                self.face_embeddings, self.num_neighbours, "dot_product"
            )
            .tree(
                num_leaves=self.num_leaves,
                num_leaves_to_search=self.num_leaves_to_search,
                training_sample_size=len(self.face_embeddings),
            )
            .score_ah(
                2,
                anisotropic_quantization_threshold=self.anisotropic_quatization_thrshld,
            )
            .reorder(self.face_embeddings.shape[-1])
            .build()
        )

        # save model and data
        if self.overwrite_model:
            logger.info(f"saving scann model to {self.model_path}")
            self.searcher.serialize(self.model_path)
            with open(self.face_image_names_path, "wb") as file:
                pickle.dump(self.face_image_names, file)
        else:
            logger.warning(f"Not saving scann model")

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

            # knn serach using scann model
            neighbors, distances = self.searcher.search_batched(
                face_embedding, number_of_images
            )
            face_names_index = neighbors[0].astype("int").astype("str")
            similar_faces_names = list(map(self.face_image_names.get, face_names_index))

            for k in zip(neighbors[0], similar_faces_names, distances[0]):
                logger.debug(f"index, name, distance : {k}")

            return similar_faces_names

    def show_images(
        self, ref_image: str, similar_faces_names: List[str], base_path: str
    ):
        """function to dispaly the input and mathched images

        Args:
            ref_image (str): path tot he input image for searching
            similar_faces_names (List[str]): names of the similar/matched face images
            base_path (str): base path to the image directory
        """
        number_of_images = len(similar_faces_names) + 1
        cols = 4
        rows = np.ceil(number_of_images / cols)
        frame = cv2.imread(ref_image)
        plt.figure(figsize=(12, 8))
        plt.subplot(rows, cols, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("input image")
        plt.axis("off")
        for k in range(1, number_of_images):
            plt.subplot(rows, cols, k + 1)
            frame = cv2.imread(os.path.join(base_path, similar_faces_names[k - 1]))
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
        plt.tight_layout()
        plt.show()
