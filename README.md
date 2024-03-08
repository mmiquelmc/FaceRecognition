# Face Recognition with OpenCV and face_recognition

This repository contains a Python script for real-time face recognition using OpenCV and the face_recognition library. The script captures video from a camera and identifies faces based on known face encodings.

## Features

- Real-time face recognition
- Displays the name and confidence level of recognized faces
- Handles multiple faces in the frame

## Requirements

- Python 3.x
- OpenCV
- face_recognition
- NumPy

## Installation

- install dependencies

```bash
pip install opencv-python
pip install face_recognition
```

## Usage

- Place your known face images in the faces directory.
- Run the script:
    
    ```bash
    python main.py
    ```
- Press `q` to exit the program.

## Customization

- Modify the size of the rectangles and text displayed on recognized faces by adjusting the parameters in the run_recognition method.

## Acknowledgements

- Coded following the tutorial made by [Indently](https://www.youtube.com/@Indently)
- [Video link](https://www.youtube.com/watch?v=tl2eEBFEHqM&ab_channel=Indently)