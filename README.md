
# Face Recognition Attendance System

This is a Python script that utilizes face recognition to mark attendance by detecting and recognizing faces in a webcam feed.

## Features

- Detects and recognizes faces in real-time using a webcam.
- Compares the detected faces with pre-encoded face images for identification.
- Marks attendance by recording the names of recognized individuals and the timestamp of their detection.
- Stores attendance records in a text file.

## Requirements

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- face_recognition (`pip install face_recognition`)

## Usage

1. Clone the repository or download the script file.
2. Install the required packages by running `pip install -r requirements.txt`.
3. Place the face images of individuals in the `image` folder. Each image file should correspond to a single person and be named with the person's name (e.g., `john.jpg`, `emma.png`).
4. Run the script using the command `python face_recognition_attendance.py`.
5. The webcam feed will open, and the script will start recognizing faces and marking attendance.
6. Press 'q' to quit the program.

## Configuration

- Modify the `path` variable in the script to specify the path to the folder containing the face images.
- The attendance records are stored in the `att.txt` file. You can modify the filename or path in the `mark_attendance()` function if desired.

## Limitations

- The accuracy of face recognition depends on the quality of the face images and the variation in lighting, angles, and occlusions.
- It is recommended to have sufficient lighting and clear images of individuals for better recognition results.

## Acknowledgments

This script utilizes the following libraries:

- OpenCV: https://opencv.org/
- NumPy: https://numpy.org/
- face_recognition: https://github.com/ageitgey/face_recognition

## License

[MIT License](LICENSE)

Feel free to modify and use this code according to your needs.

If you have any suggestions or encounter any issues, please open an issue or submit a pull request.

## Author

[Rocker Ritesh](sumityadav.com.np)