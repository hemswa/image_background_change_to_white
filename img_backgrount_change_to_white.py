import cv2
from PIL import Image, ImageOps

def process_image(input_file, output_file):
    # Load the image using OpenCV
    img = cv2.imread(input_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Load a pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        print("No face detected. Exiting.")
        return
    
    # Assume the first detected face is the main face
    x, y, w, h = faces[0]
    
    # Crop the image to the face region with some padding
    padding = max(w, h) // 2  # Adjust as needed
    x_start = max(x - padding, 0)
    y_start = max(y - padding, 0)
    x_end = min(x + w + padding, img.shape[1])
    y_end = min(y + h + padding, img.shape[0])
    
    cropped_face = img[y_start:y_end, x_start:x_end]
    
    # Convert to PIL Image for further processing
    pil_image = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
    
    # Create a white background canvas
    canvas_size = max(cropped_face.shape[:2]) + padding * 2
    white_bg = Image.new("RGB", (canvas_size, canvas_size), "white")
    
    # Center the face on the canvas
    offset = ((canvas_size - pil_image.width) // 2, (canvas_size - pil_image.height) // 2)
    white_bg.paste(pil_image, offset)
    
    # Save the result
    white_bg.save(output_file)
    print(f"Processed image saved as {output_file}")

# Example usage
input_image = input("Enter input image file (e.g., input_img.jpeg): ").strip()
output_image = input("Enter output image file (e.g., output_img.jpeg): ").strip()
process_image(input_image, output_image)