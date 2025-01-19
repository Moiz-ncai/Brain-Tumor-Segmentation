import cv2
from ultralytics import YOLO

# Load a model
model = YOLO("best.pt")  # trained YOLOv11 model


def run_inference(image_path):
    """
    Run inference on a single image using YOLOv11 and display the result using OpenCV.
    :param image_path: Path to the input image.
    """
    # Run inference
    results = model(image_path)  # return a list with a single Results object

    # Process the result
    result = results[0]  # Get the single result object

    # Load the original image
    image = cv2.imread(image_path)

    # Create a semi-transparent overlay
    overlay = image.copy()

    # Draw the segmentation masks
    if hasattr(result, "masks") and result.masks is not None:
        for mask in result.masks.data:
            # Convert mask to NumPy array
            mask_np = mask.cpu().numpy()  # Convert the mask to NumPy format
            mask_np = (mask_np > 0.5).astype("uint8") * 255  # Binarize the mask

            # Find contours from the mask
            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Fill the contours with a semi-transparent color and draw blue outlines
            for contour in contours:
                cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)  # Fill
                cv2.drawContours(image, [contour], -1, (255, 0, 0), thickness=2)  # Blue outline

                # Calculate contour area in pixels
                area_pixels = cv2.contourArea(contour)
                if area_pixels > 0:  # Avoid tiny noise contours
                    # Convert pixel area to cm²
                    area_cm2 = area_pixels / 1736.6  # Conversion factor
                    x, y, w, h = cv2.boundingRect(contour)
                    label = f"Area: {area_cm2:.2f} cm²"
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Blend the overlay with the original image
        alpha = 0.5  # Transparency factor
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    else:
        print("No segmentation masks detected.")

    # Display the result using OpenCV
    cv2.imshow("YOLO Inference", image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()  # Close the OpenCV window


if __name__ == "__main__":
    # Parameters
    image_path = "test/images/y2_jpg.rf.d300137736af9b9814ec1d4c2a84f07e.jpg"  # Replace with your image path

    # Run inference
    run_inference(image_path)
