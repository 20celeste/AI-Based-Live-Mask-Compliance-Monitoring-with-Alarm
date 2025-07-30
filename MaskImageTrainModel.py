import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO(r"C:\Users\keert\Downloads\best(mask).pt")

# Load image (replace with your image path)
image_path = "C:/Image/mask.jpg"
image = cv2.imread(image_path)
# Check if image was loaded
if image is None:
    print("Failed to load image.")
else:
    # Run YOLOv8 on the image
    results = model(image, verbose=False)

    # Draw results
    annotated_image = results[0].plot()

    # Display result
    cv2.imshow("YOLOv8 Detection on Image", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()