# faceSearch
A face searching AI application using face detection and face embedding 

The algorithms works as follows:
1. Detect face bounding boxes from input images
2. Crop detected face region and compute 128 dimension face embeddings.
3. Train a ScaNN model for approximate KNN neighbours

For searching similar faces from the database use the model in step 3 to find the approximate nearest neighbours.

## Training a face searcher 
```
python3 run_face_search.py train_face_searcher --model_dir=<path where the model needs to be saved> --image_database_dir=<directory containing training images (jpg/png)>

```

## Searching similar faces from the database
```
python3 run_face_search.py search_similar_faces --input_image_file=<path to input image file> --number_of_images=<number of images to be retrieved from the database>--model_dir=<path to trained model directory> --image_database_dir=<directory containing training/database images (jpg/png)>
```

## An example of searching using celebrity database
first row is the query/input image
rest of the rows are retrieved similar faces images
![fig1](https://user-images.githubusercontent.com/32892726/198875336-7d1c4e5f-9cdc-44fa-acdf-e57ffc14d792.png)
