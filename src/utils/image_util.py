import base64
from PIL import Image, ImageDraw
import re


def load_image(inputs: dict) -> dict:
    """Load image from file and encode it as base64."""

    image_path = inputs["image_path"]

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_base64 = encode_image(image_path)
    return {"image": image_base64}


def resize_image(image_path, output_path, max_width=950, max_height=630):
    """Resize the image to the maximum allowed resolution of 950x630."""

    with Image.open(image_path) as img:
        # check if the image size exceeds the maximum allowed resolution
        if img.width > max_width or img.height > max_height:
            # calculate the new size preserving the aspect ratio
            aspect_ratio = img.width / img.height
            if aspect_ratio > 1:  # width is greater than height
                new_width = max_width
                new_height = int(max_width / aspect_ratio)
            else:  # height is greater than width or they are equal
                new_height = max_height
                new_width = int(max_height * aspect_ratio)

            # resize the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            # save the resized image
            resized_img.save(output_path)

        else:
            # If the image is within the desired resolution, just save it to the output path
            img.save(output_path)


def draw_bounding_boxes(output_path: str, bounding_boxes: str):
    """Draw bounding boxes on the image."""

    image = Image.open(output_path)
    draw = ImageDraw.Draw(image)

    # extract numeric values using regular expressions
    numeric_values = re.findall(r"\d+", bounding_boxes)

    if len(numeric_values) % 4 != 0:
        print("Invalid bounding box format")
        return

    # convert extracted values to integers
    numeric_values = list(map(int, numeric_values))

    # iterate through the numeric values and draw the bounding boxes
    for i in range(0, len(numeric_values), 4):
        x1, y1, x2, y2 = numeric_values[i : i + 4]
        box_coordinates = (x1, y1, x2, y2)
        draw.rectangle(box_coordinates, outline="red", width=3)

    image.show()
