# LBPH Face Recognition System

This project uses the **LBPH (Local Binary Pattern Histograms)** face recognition algorithm to identify individuals. It consists of two main parts:
1. **Training Process**: A model is trained using a face dataset to recognize faces.
2. **Testing and Recognition Process**: The trained model is used to recognize faces in a test image, and the results are displayed on the screen.

## Technologies Used
- Python
- OpenCV (Face Detection and Recognition)
- Numpy

## Installation

1. **Install Required Libraries**:
   ```bash
   pip install opencv-python numpy
   ```

2. **Training Data**:
   - Download the LFW (Labeled Faces in the Wild) dataset and place it in the `lfw-deepfunneled` folder.
   - You can download the LFW dataset [here](https://www.kaggle.com/datasets/dbeley/lfw).

3. **Training**:
   To train the face recognition model, run the following command:
   ```bash
   python train_face_recognizer.py
   ```

   This will save the trained model as `face_recognizer.yml`.

4. **Testing**:
   To test the trained model, run the following command:
   ```bash
   python test_face_recognition.py
   ```

## File Structure
- `train_face_recognizer.py`: Python file for training the face recognition model.
- `test_face_recognition.py`: Python file for testing the face recognition model.
- `face_recognizer.yml`: The trained face recognition model.
- `lfw-deepfunneled/`: Folder containing images from the LFW dataset.

## Usage

1. Once the model is trained, you can use it to recognize faces in test images.
2. During the testing phase, the recognized person’s name and confidence level will be displayed.

## Face Recognition Results

When the face recognition process is successful, you should see the following information:
- **Recognized Person**: The person identified by the model.
- **Confidence Level**: The confidence level of the model's prediction.
