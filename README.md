# Bird vs Drone Detection and Classification

#### Aim : To develop a deep learning-based solution that can classify aerial images into two categories Bird or Drone and optionally perform object detection (YOLO) to locate and label these objects in real-world scenes.

## Project Takeaways:
- Deep Learning
- Computer Vision
- Image Classification & Object Detection
- Python
- TensorFlow/Keras 
- Data Preprocessing & Augmentation
- YOLOv8 
- Model Evaluation
- Streamlit Deployment

## WorkFlow
### - Image Classification Training (Custom and Transfer Learning Model) [Aerial_Object_Detection.ipynb](https://github.com/kevinxaviour/Aerial_Object_Classification_and_Detection/blob/b0ab59990be0693f9afe882f00639cfad5e649c3/Aerial_Object_Detection.ipynb)
  - Used cv2 to show an example on Data augmentation
  - Used ImageDataGenerator library to create augmented data in training_data
    - Resize, rotation, zoom, horizontal view, vertical view
  - Data Preprocessing
    - Resize images to (224,224) pixels
  - Custom Training from Scratch
  - Introducing Transfer Learning with 3 different pre-trained applications from tensorflow.keras.applications
  - Model Evaluation
  - Chose the best model among the 4 and saved the model for future use.

  
    <img width="321" height="350" alt="image" src="https://github.com/user-attachments/assets/bad62cbf-6b6e-45db-b6ea-1f055a1500f7" />

### - Image Detection Training (YOLO Model) [YoloModel.ipynb](https://github.com/kevinxaviour/Aerial_Object_Classification_and_Detection/blob/b0ab59990be0693f9afe882f00639cfad5e649c3/YoloModel.ipynb)
- Data Preperation
  - Yolo requires the training images to be in an particular folder dir with train,validation and test data split in labels and images folders seperately.
- Creating data.yaml file for model training [data.yaml](https://github.com/kevinxaviour/Aerial_Object_Classification_and_Detection/blob/b0ab59990be0693f9afe882f00639cfad5e649c3/data.yaml)
- Model Training
- Model Evaluation

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/901b8434-de3a-4f7d-be59-4d8474b758b4" />

- Saving Model

### - Streamlit Application [streamlit.py](https://github.com/kevinxaviour/Aerial_Object_Classification_and_Detection/blob/b0ab59990be0693f9afe882f00639cfad5e649c3/streamlit.py)
-  Side bar to upload data and select a model to view result.
-  In main content the left side shows original image and right side with the predicted output.
-  Created an download button to download YOLO detected image
### -  Image Classification

<img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/f523aab7-4bd9-46c6-8ef5-ba7b1427e130" />

### - Image detection

 <img width="800" height="600" alt="image" src="https://github.com/user-attachments/assets/4d565872-4c7f-4554-9b6e-650d8d698286" />



