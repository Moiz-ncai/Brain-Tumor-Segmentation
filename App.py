import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QHBoxLayout, \
    QWidget, QLineEdit, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO


class TumorDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Brain Tumor Detection")
        self.setGeometry(100, 100, 1100, 900)

        self.image_path = None
        self.pixel_to_cm_scale = None
        self.tumor_size = "N/A"

        self.initUI()

        # Load the YOLO model
        self.model = YOLO("best.pt")  # Replace with your trained YOLOv11 model

    def initUI(self):
        # Set dark mode style for the window
        self.setStyleSheet("background-color: #121212; color: white;")

        # Main layout
        main_layout = QVBoxLayout()

        # Top bar for tumor size
        top_bar = QHBoxLayout()
        self.tumor_size_label = QLabel(f"Tumor size: {self.tumor_size}", self)
        self.tumor_size_label.setAlignment(Qt.AlignCenter)
        self.tumor_size_label.setStyleSheet("font-size: 30px; font-weight: bold; color: white; padding: 10px;")
        top_bar.addWidget(self.tumor_size_label)
        main_layout.addLayout(top_bar)

        # Central widget for the image display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 1px solid white; background-color: #1E1E1E;")
        self.image_label.setFixedSize(800, 800)
        main_layout.addWidget(self.image_label)

        # Right bar for controls
        right_bar = QVBoxLayout()
        right_bar.setSpacing(50)  # Increase spacing between widgets to 50px

        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)
        self.upload_button.setFixedSize(170, 50)  # Increased size
        self.upload_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                font-weight: bold; 
                background-color: #007BFF; 
                color: white; 
                padding: 5px; 
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #0056b3; /* Darker blue for hover */
            }
            QPushButton:pressed {
                background-color: #003f7f; /* Even darker blue for pressed */
            }
        """)
        right_bar.addWidget(self.upload_button, alignment=Qt.AlignHCenter)

        # Create a vertical layout for the scale label and input
        scale_layout = QVBoxLayout()
        scale_layout.setSpacing(5)  # Reduce spacing between the label and input

        scale_label = QLabel("Pixel-to-cm Scale:", self)
        scale_label.setStyleSheet("font-size: 16px; color: white; font-weight: bold;")
        scale_layout.addWidget(scale_label, alignment=Qt.AlignCenter)

        self.scale_input = QLineEdit(self)
        self.scale_input.setText("41.68")  # Autofill with the average scale
        self.scale_input.setStyleSheet(
            "font-size: 16px; background-color: #1E1E1E; color: white; padding: 5px; border: 1px solid white; border-radius: 3px;"
        )
        self.scale_input.setFixedWidth(100)
        scale_layout.addWidget(self.scale_input, alignment=Qt.AlignCenter)

        # Add the scale layout to the right bar
        right_bar.addLayout(scale_layout)

        self.run_button = QPushButton("Run Segmentation", self)
        self.run_button.clicked.connect(self.run_segmentation)
        self.run_button.setEnabled(False)  # Initially disabled
        self.run_button.setFixedSize(170, 50)
        self.run_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                font-weight: bold; 
                background-color: #555555;  /* Gray for disabled state */
                color: #AAAAAA;  /* Light gray for text in disabled state */
                padding: 5px; 
                border-radius: 5px;
            }
            QPushButton:enabled {
                background-color: #007BFF;  /* Blue for enabled state */
                color: white;
            }
            QPushButton:enabled:hover {
                background-color: #0056b3;  /* Darker blue for hover */
            }
            QPushButton:enabled:pressed {
                background-color: #003f7f;  /* Even darker blue for pressed */
            }
        """)
        right_bar.addWidget(self.run_button, alignment=Qt.AlignHCenter)

        # Add a spacer above and below the controls
        spacer_top = QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        spacer_bottom = QSpacerItem(20, 100, QSizePolicy.Minimum, QSizePolicy.Expanding)
        right_bar.insertSpacerItem(0, spacer_top)
        right_bar.addSpacerItem(spacer_bottom)

        # Horizontal layout to include the right bar
        horizontal_layout = QHBoxLayout()
        horizontal_layout.addLayout(main_layout, stretch=3)  # Increase stretch for main content
        right_widget = QWidget()
        right_widget.setLayout(right_bar)
        right_widget.setFixedWidth(200)  # Adjusted width for thinner right bar
        horizontal_layout.addWidget(right_widget, stretch=1)  # Decrease stretch for right bar

        # Main widget
        container = QVBoxLayout()
        container.addLayout(horizontal_layout)

        # Footer
        footer = QLabel("Project by Abdul Moiz Qarni - 17PWMCT0564", self)
        footer.setAlignment(Qt.AlignCenter)
        footer.setStyleSheet("font-size: 14px; color: #BBBBBB; padding: 10px;")
        container.addWidget(footer)

        main_widget = QWidget()
        main_widget.setLayout(container)
        self.setCentralWidget(main_widget)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Image Files (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_path:
            self.image_path = file_path
            self.display_image(self.image_path)
            self.run_button.setEnabled(True)

    def display_image(self, path):
        image = QPixmap(path)
        self.image_label.setPixmap(image.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def run_segmentation(self):
        try:
            self.pixel_to_cm_scale = float(self.scale_input.text())
        except ValueError:
            self.scale_input.setText("Invalid scale value!")
            return

        if not self.image_path:
            return

        # Load the image and run YOLO inference
        results = self.model(self.image_path)
        result = results[0]  # Single result object

        # Load the original image
        image = cv2.imread(self.image_path)
        overlay = image.copy()

        total_area_cm2 = 0  # Initialize total area

        if hasattr(result, "masks") and result.masks is not None:
            for mask in result.masks.data:
                mask_np = mask.cpu().numpy()  # Convert to NumPy
                mask_np = (mask_np > 0.5).astype("uint8") * 255

                # Find contours
                contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # Fill
                    cv2.drawContours(image, [contour], -1, (255, 0, 0), thickness=3)  # Blue outline

                    # Calculate tumor area in pixels and convert to cm²
                    area_pixels = cv2.contourArea(contour)
                    area_cm2 = area_pixels / (self.pixel_to_cm_scale ** 2)
                    total_area_cm2 += area_cm2  # Accumulate total area

            # Blend overlay
            alpha = 0.5
            image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

            # Update the tumor size in the top bar
            self.tumor_size = f"{total_area_cm2:.2f} cm²"
            self.tumor_size_label.setText(f"Tumor size: {self.tumor_size}")

        # Convert image to QImage and display
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        self.image_label.setPixmap(
            QPixmap.fromImage(q_image).scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TumorDetectionApp()
    window.show()
    sys.exit(app.exec_())
