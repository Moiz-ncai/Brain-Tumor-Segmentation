# Brain Tumor Detection App

## Overview
The **Brain Tumor Detection App** is a desktop application built using **Python** and **PyQt5** for detecting brain tumors in medical images. It utilizes a **YOLO (You Only Look Once)** deep learning model for image segmentation and overlays detected tumor areas on the input image. The application also estimates the tumor size in **square centimeters (cm²)**.

## Features
- **Upload Medical Images**: Supports common image formats like PNG, JPG, JPEG, and BMP.
- **Automatic Tumor Detection**: Uses a trained YOLO model (`best.pt`) to segment tumor areas.
- **Tumor Size Estimation**: Calculates the tumor area in cm² based on a user-defined pixel-to-cm scale.
- **User-Friendly Interface**: Dark-mode themed UI with interactive controls.
- **Image Overlay**: Highlights detected tumor areas with contours and overlays.

## Installation
### Prerequisites
Ensure you have **Python 3.8+** installed on your system. You also need to install the following dependencies:

```bash
pip install -r requirements.txt
```

### Clone the Repository
```bash
git clone https://github.com/your-repository/tumor-detection-app.git
cd tumor-detection-app
```

## Usage
### Running the Application
Run the following command:
```bash
python app.py
```

### Steps to Use
1. **Upload Image**: Click the `Upload Image` button and select a medical image.
2. **Set Scale**: Adjust the pixel-to-cm scale if necessary.
3. **Run Segmentation**: Click `Run Segmentation` to analyze the image.
4. **View Results**: The application will display the tumor area in cm² and overlay the detected region.

## Future Improvements
- Support for **multiple tumor detection** in a single image.
- Enhance **model accuracy** with additional training data.
- Develop a **web-based version** for remote access.

## License
This project is open-source under the **MIT License**.

## Author
Developed by **Abdul Moiz Qarni**.

---
**Note:** Ensure that `best.pt` is placed in the project directory before running the application.

