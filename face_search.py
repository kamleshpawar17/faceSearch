import pickle
from pathlib import Path
from typing import Any, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from lshashing import LSHRandom
from tqdm import tqdm

from face_detection import detectFace
from face_embedding import getFaceEmbeddings


class faceSearch:
    def __init__(self, config: dict) -> None:
        self.model_path = Path(config["LSH_MODEL"])
        self.embeddings_path = Path(config["FACE_EMBEDDINGS"])
        self.image_types = config["IMAGE_TYPES"]
        self.hash_len = config["HASH_LENGTH"]
        self.num_tables = config["NUMBER_OF_TABLES"]
        self.face_image_names_path = config["FACE_IMAGE_NAMES"]
        self.buckets = config["BUCKETS"]
        self.radius = config["RADIUS"]
        self.face_detector = detectFace(config)
        self.face_embedder = getFaceEmbeddings(config)
        if self.model_path.exists() and self.embeddings_path.exists():
            with open(self.model_path, "rb") as file:
                self.lsh_model = pickle.load(file)
            with open(self.face_image_names_path, "rb") as file:
                self.face_image_names = pickle.load(file)
            with open(self.embeddings_path, "rb") as file:
                self.embeddings = pickle.load(file)
        else:
            self.lsh_model = None

        # self.lsh_model = None

    def register_faces_from_dir(self, input_dir: str) -> None:
        # get face embeddings
        (
            face_embeddings,
            face_image_names,
        ) = self.face_embedder.get_face_embeddings_from_dir(input_dir)

        # create the LSH model
        if self.lsh_model is None:
            self.embeddings = face_embeddings
            self.face_image_names = face_image_names
            self.lsh_model = LSHRandom(
                face_embeddings, hash_len=self.hash_len, num_tables=self.num_tables
            )
        else:
            self.embeddings = np.concatenate((self.embeddings, face_embeddings), axis=0)
            self.face_image_names.extend(face_image_names)
            self.lsh_model.add_new_entry(face_embeddings)

        # save model and data
        with open(self.model_path, "wb") as file:
            pickle.dump(self.lsh_model, file)
        with open(self.embeddings_path, "wb") as file:
            pickle.dump(self.embeddings, file)
        with open(self.face_image_names_path, "wb") as file:
            pickle.dump(self.face_image_names, file)

    def search_similar_faces(self, input_image: str, number_of_images: int) -> Any:
        # read image
        frame = cv2.imread(input_image)
        # detect faces
        face_rects = self.face_detector.detect_face(frame)
        # compute face embeddings
        if len(face_rects) > 0:
            face_embedding = np.squeeze(
                np.array(self.face_embedder.get_128d_face_embeddings(frame, face_rects))
            )
            if len(face_embedding.shape) == 1:
                face_embedding = np.expand_dims(face_embedding, 0)
            # knn serach using LSH model
            nearest_faces = np.array(
                self.lsh_model.knn_search(
                    self.embeddings,
                    face_embedding[0, :],
                    k=number_of_images,
                    buckets=self.buckets,
                    radius=self.radius,
                )
            )
            face_names_index = nearest_faces[:, 1].astype("int")
            similar_faces_names = list(np.take(self.face_image_names, face_names_index))

            for table in self.lsh_model.tables:
                logger.debug(f"self.lsh_model:\n {table.hash_table}")
            logger.debug(f"self.embeddings: {self.embeddings.shape}")
            logger.debug(f"face_image_names: {len(self.face_image_names)}")
            logger.debug(f"nearest_faces:\n {nearest_faces}")
            logger.debug(f"selected images:\n {similar_faces_names}")

            return similar_faces_names

    def show_images(self, ref_image: str, similar_faces_names: List[str]):
        number_of_images = len(similar_faces_names) + 1
        cols = 4
        rows = np.ceil(number_of_images / cols)
        frame = cv2.imread(ref_image)
        plt.figure(figsize=(12, 8))
        plt.subplot(rows, cols, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        for k in range(1, number_of_images):
            plt.subplot(rows, cols, k + 1)
            frame = cv2.imread(similar_faces_names[k - 1])
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.axis("off")
        plt.tight_layout()
        plt.show()
