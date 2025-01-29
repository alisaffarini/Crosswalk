
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from model import LYTNetV2
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import MakeModel  # This imports your MakeModel class from model.py

# Load the model
def load_model(state_dict_path):
    model = MakeModel(pretrained=False)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Prepare the image
def prepare_image(image_path, output_size=[768, 576]):
    image = Image.open(image_path)
    image = image.resize((output_size[0], output_size[1]), Image.BILINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255)
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image

# Run inference
def run_inference(model, image):
    with torch.no_grad():
        output = model(image)
    return output

# Process output
def process_output(output):
    coordinates = output['coordinates'].squeeze().tolist()
    is_crosswalk = output['IScrosswalk'].squeeze().item()
    return coordinates, is_crosswalk


import cv2
import numpy as np

def visualize_coordinates(image_path, coordinates, output_path):
    # Read the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Denormalize and convert coordinates to integers
    x1, y1, x2, y2 = [int(coord * width if i % 2 == 0 else coord * height) for i, coord in enumerate(coordinates)]

    # Draw the bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw the coordinate points
    cv2.circle(img, (x1, y1), 5, (255, 0, 0), -1)
    cv2.circle(img, (x2, y2), 5, (0, 0, 255), -1)

    # Add labels
    cv2.putText(img, "Top-left", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.putText(img, "Bottom-right", (x2, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Visualization saved to {output_path}")


def main():
    # Load the model
    model = load_model('moreefficientweights.pth')

    # List of test images
    test_images = ['trial.jpg']
    for image_path in test_images:
        # Prepare the image
        image = prepare_image(image_path)

        # Run inference
        output = run_inference(model, image)

        # Process and print the results
        coordinates, is_crosswalk = process_output(output)
        output_path = 'trialtest2.jpg'
        visualize_coordinates(image_path, coordinates, output_path)

        print(f"Results for {image_path}:")
        print(f"Coordinates: {coordinates}")
        print(f"Is crosswalk: {is_crosswalk:.2f}")
        #print(f"Light class: {light_class}")
        print("--------------------")

if __name__ == "__main__":
    main()