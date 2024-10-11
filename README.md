# 3D Reconstruction and Plane Detection using Epipolar Geometry

Welcome to the **3D Reconstruction and Plane Detection** project! This repository contains a Python script that performs 3D reconstruction from stereo images using epipolar geometry. It detects keypoints, matches them between images, computes the Fundamental and Essential matrices, reconstructs a 3D point cloud, and fits planes to the reconstructed points using RANSAC.

<p align="center">
  <img src="https://github.com/user-attachments/assets/639179dd-c09f-4a3c-be43-d0c1c408ea8f" alt="Detected Planes with Vector" width="600">
</p>

## Introduction

This project demonstrates how to perform:

- 🔑 **Keypoint detection using SIFT**
- 🔗 **Feature matching between two images**
- 📐 **Computation of the Fundamental and Essential matrices**
- 🧭 **Recovery of camera pose (rotation and translation)**
- 🌐 **Triangulation of matched points to create a 3D point cloud**
- ✈️ **Plane fitting using RANSAC**
- 🎨 **Visualization of keypoints, matches, epipolar lines, and planes**

## Features

- **Keypoint Detection**: Detects and computes descriptors using SIFT.
- **Feature Matching**: Matches features using BFMatcher with a ratio test.
- **Epipolar Geometry**: Calculates the Fundamental and Essential matrices.
- **Pose Recovery**: Determines the relative camera positions.
- **3D Reconstruction**: Triangulates points to form a 3D point cloud.
- **Plane Detection**: Uses RANSAC to detect planes in the point cloud.
- **Visualization**: Provides comprehensive plotting functions for analysis.

## Requirements

- 🐍 **Python 3.6 or higher**
- 📦 **NumPy**
- 👁️ **OpenCV (cv2)**
- 📊 **Matplotlib**

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/danielbob32/ComputerVisonImageMatching
   cd ComputerVisonImageMatching
   ```
2. **Install Dependencies**

Install the required Python packages:
```bash
pip install numpy opencv-python matplotlib
```
## Usage
1. **Prepare Your Data**

- 📁 Create a directory for your data, e.g., data/example_3/.
- 🖼️ Place your stereo images I1.png and I2.png in this directory.
- 📄 Provide the camera intrinsic matrix K.txt in the same directory. The matrix should be in text format with comma-separated values.

2. **Configure Parameters**
In the main() function of the script, you can adjust the following parameters:
```python
data_dir = 'data/example_3'  # Path to your data directory
amount = 70                  # Number of matches and epipolar lines to display
planes_amount = 2            # Number of planes to detect
Ransac_iterations = 1000     # RANSAC iterations for plane fitting
threshold = 0.5              # Distance threshold for RANSAC
```
3. **Run the Script**

Execute the script using the command line:
```bash
python main.py
```
4. **View the Results**

The script will display various plots:

- 📷 Keypoints detected in both images
- 🔍 Matched features between images
- 📏 Matches with connecting lines
- 📐 Epipolar lines corresponding to the matches
- 🛰️ Planes fitted to the 3D point cloud
- 🧭 Normals of the detected planes projected onto the images
  
## Example Results
### Keypoint Detection
<p align="center"> <img src="https://github.com/user-attachments/assets/e13fbf73-0f07-41c2-9a3b-cdef1cf6f6c8" alt="Keypoint Detection" width="600"> </p>

### Feature Matching
<p align="center"> <img src="https://github.com/user-attachments/assets/4944b31f-5f93-45a4-a85a-0d91df5c5c84" alt="Feature Matching" width="600"> </p>

### Epipolar Lines
<p align="center"> <img src="https://github.com/user-attachments/assets/5820235a-10ce-4d78-90ab-c1461cd84505" alt="Epipolar Lines" width="600"> </p>

### Plane Detection
<p align="center"> <img src="https://github.com/user-attachments/assets/521da56d-ab54-4388-92f5-89af64c0373f" alt="Plane Detection" width="600"> </p>

### Normals Projection
<p align="center"> <img src="https://github.com/user-attachments/assets/b706edb1-90fd-4daa-9a21-f19868265165" alt="Normals Projection" width="600"> </p>

Note: The results directory should contain the generated images from running the script.

## Project Structure
```
- main.py: Main Python script containing all functions and the main() function.
- data/: Directory containing sample images and camera matrix.
- results/: Directory where output images and plots are saved.
- README.md: Project documentation.
```
## Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

1. Fork the repository.
2. Create your feature branch: git checkout -b feature/YourFeature
3. Commit your changes: git commit -am 'Add some feature'
4. Push to the branch: git push origin feature/YourFeature
5. Open a pull request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or suggestions, please contact Daniel Bobritski:

📧 Email: danielbob32@gmail.com
💼 LinkedIn: Daniel Bobritski
🎮 Discord: danielxp13#9709
