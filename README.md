# 3D Reconstruction and Plane Detection using Epipolar Geometry

Welcome to the **3D Reconstruction and Plane Detection** project! This repository contains a Python script that performs 3D reconstruction from stereo images using epipolar geometry. It detects keypoints, matches them between images, computes the Fundamental and Essential matrices, reconstructs a 3D point cloud, and fits planes to the reconstructed points using RANSAC.

![filtered matching points based on epi lines](https://github.com/user-attachments/assets/af670775-93b9-4745-93a1-d3145ce1efac)
![Detected Planes with vector](https://github.com/user-attachments/assets/639179dd-c09f-4a3c-be43-d0c1c408ea8f)

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

📷 Keypoints detected in both images
🔍 Matched features between images
📏 Matches with connecting lines
📐 Epipolar lines corresponding to the matches
🛰️ Planes fitted to the 3D point cloud
🧭 Normals of the detected planes projected onto the images
Example Results
Keypoint Detection

Feature Matching

Epipolar Lines

Plane Detection

Normals Projection

Note: The results directory should contain the generated images from running the script.

Project Structure
your_script_name.py: Main Python script containing all functions and the main() function.
data/: Directory containing sample images and camera matrix.
I1.png: First stereo image.
I2.png: Second stereo image.
K.txt: Camera intrinsic matrix.
results/: Directory where output images and plots are saved.
README.md: Project documentation.
Contributing
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.

Fork the repository.
Create your feature branch: git checkout -b feature/YourFeature
Commit your changes: git commit -am 'Add some feature'
Push to the branch: git push origin feature/YourFeature
Open a pull request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For questions or suggestions, please contact Daniel Bobritski:

📧 Email: danielbob32@gmail.com
💼 LinkedIn: Daniel Bobritski
🎮 Discord: danielxp13#9709
