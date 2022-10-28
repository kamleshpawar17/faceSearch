import yaml
from loguru import logger

from face_search import faceSearch

log_level = "INFO"
# os.environ["LOGURU_LEVEL"] = log_level
logger.add("logs/logs.log", level=log_level)


if __name__ == "__main__":
    # ----- read config refer to config.yaml for description of parameters---- #
    logger.info("Loading configuration")
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # detect face class
    # face embedding class
    # search class -> register, search,
    face_search = faceSearch(config)
    input_dir = "input_images/"
    # input_image = "input_images/shivani_pp_one_photo.jpg"
    input_image = "input_images/kamlesh_pp_one_photo.png"
    # input_image = "input_images/kiara_pp_2022_2inx2in.png"
    # face_search.register_faces_from_dir(input_dir)
    similar_faces_names = face_search.search_similar_faces(input_image, 20)
    face_search.show_images(input_image, similar_faces_names)
