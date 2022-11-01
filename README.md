# faceSearch
A face searching AI application using face detection and face embedding 

The algorithms works as follows:
1. Detect face bounding boxes from input images
2. Crop detected face region and compute 128 dimension face embeddings.
3. Train a HNSW model for approximate KNN neighbours

For searching similar faces from the database use the model in step 3 to find the approximate nearest neighbours.

## Training a face searcher 
```
python3 run_face_search.py train_face_searcher --image_database_dir=<directory containing training images (jpg/png)>

```
you can modify/optimize the the parameters in ```config.yaml``` for trianing the model and paths for saving the model.

## Searching similar faces from the database
```
python3 run_face_search.py search_similar_faces --input_image_file=<path to input image file> --number_of_images=<number of images to be retrieved from the database>--model_dir=<path to trained model directory> --image_database_dir=<directory containing training/database images (jpg/png)>
```

## An example of searching using celebrity database
first row is the query/input image and rest of the rows are retrieved similar faces images


![fig1](https://user-images.githubusercontent.com/32892726/199361706-979cf8c5-929e-4b50-9dce-be6b1eb8dfc2.png)


## Dependencies
```
hnswlib
opencv-python
loguru
yaml
fire
```
