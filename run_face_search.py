from loguru import logger
import yaml
import fire

from face_search import faceSearch


def train_face_searcher(image_database_dir: str, model_dir: str):
    """function to train a face searcher using an image database

    Args:
        image_database_dir (str): path to image database ie. a directory containing jpg/png images
        model_dir (str): path to a directory where a trained scann model will be saved
    """
    logger.info("Loading configuration for running train_face_searcher()")
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["SCANN_MODEL"] = model_dir

    face_search = faceSearch(config)
    face_search.register_faces_from_dir(image_database_dir)


def search_similar_faces(
    input_image_file: str,
    number_of_images: int,
    model_dir: str,
    image_database_dir: str,
):
    """function to query scann model and retrieve similar images from the database

    Args:
        input_image_file (str): path to input image file
        number_of_images (int): number of similar images to be retrieved from the database
        model_dir (str): path to scann model directory
        image_database_dir (str): path to image database containing images (jpg/png)
    """
    logger.info("Loading configuration for searching similar faces")
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["SCANN_MODEL"] = model_dir

    face_search = faceSearch(config)
    similar_faces_names = face_search.search_similar_faces(
        input_image_file, number_of_images
    )
    face_search.show_images(
        input_image_file, similar_faces_names, base_path=image_database_dir
    )


if __name__ == "__main__":
    fire.Fire(
        {
            "train_face_searcher": train_face_searcher,
            "search_similar_faces": search_similar_faces,
        }
    )
