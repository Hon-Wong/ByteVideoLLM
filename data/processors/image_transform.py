import numpy as np
from PIL import Image
from typing import Union, List, Tuple


class PILToNdarray:
    def __init__(self):
        pass

    def __call__(self, image: Image.Image):
        image_array = np.array(image)
        return image_array.astype(np.float32)


class Rescale:
    def __init__(self, rescale_factor: float):
        self.scale = rescale_factor

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")
        return image * self.scale


class ResizeWithAspectRatio:
    def __init__(self, size: Union[List[int], Tuple[int, int]], padding_value: float = 0, resampling=Image.BILINEAR):
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ValueError("Size must be a list or tuple with two elements: (height, width).")
        self.target_size = size
        self.padding_value = padding_value
        self.resampling = resampling

    def __call__(self, image: np.ndarray):
        if not isinstance(image, np.ndarray):
            raise NotImplementedError("Input must be a numpy array.")

        original_height, original_width = image.shape[:2]
        target_height, target_width = self.target_size

        # Calculate the new size while maintaining the aspect ratio
        aspect_ratio = original_width / original_height
        if target_width / target_height > aspect_ratio:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        else:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)

        # Resize the image
        resized_image = np.array(Image.fromarray(image.astype(np.uint8)).resize((new_width, new_height), self.resampling))

        # Create a new image with the target size and padding value
        new_image = np.full((target_height, target_width, image.shape[2]), self.padding_value, dtype=np.float32)

        # Place the resized image in the center
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        new_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

        return new_image.astype(np.float32)

# Example usage
if __name__ == "__main__":
    # Load an image using PIL
    image = Image.open('path/to/your/image.jpg')
    
    # Convert the PIL image to a NumPy array
    pil_to_ndarray = PILToNdarray()
    image_array = pil_to_ndarray(image)
    
    # Resize the image while maintaining the aspect ratio and padding with 0
    resize = ResizeWithAspectRatio(size=[300, 300], padding_value=0, resampling=Image.LANCZOS)
    resized_image = resize(image_array)
    
    # Convert back to PIL Image for visualization (optional)
    resized_image_pil = Image.fromarray(resized_image.astype(np.uint8))
    resized_image_pil.show()