import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Do not alter this path!
IMAGE_PATH: str = "data/Image01.png"


class ImageProcessor:
    def __init__(self, image_path: str, colour_type: str = "BGR"):
        """
        Load and save the provided image, the image colour type and the image directory.
        Use CV2 to load the image.

        Args:
            image_path (str): Path to the input image.
            colour_type (str): Colour type of the image (BGR, RGB, Gray).
        """
        # Extract the parent directory of the image.
        self._image_directory: str = os.path.dirname(image_path)
        if colour_type not in ["BGR", "RGB", "Gray"]:
            raise ValueError("The given colour is not supported!")

        # ToDo: Save the colour type and load the image using CV2.
        self._colour_type: str = colour_type
        if self._colour_type == "Gray":
            self._read_image = cv2.imread(image_path)
            self._read_image = cv2.cvtColor(self._read_image, cv2.COLOR_BGR2GRAY)
        if self._colour_type == "BGR":
            self._read_image = cv2.imread(image_path)
            # cv2.imshow('Image0 -- BGR', self._read_image)
        # ToDo: Hint: Using CV2, you cannot directly load RGB images. So load it in BGR and convert it using CV2.
        if self._colour_type == "RGB":
            self._read_image = cv2.imread(image_path)
            self._read_image = cv2.cvtColor(self._read_image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Image1 -- RGB', self.cvt_image)
        # #cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # bgr_pixel = self._read_image[0, 0]
        # rgb_pixel = self.cvt_image[0, 0]

        # #Ensuring that the pixel values are correct
        # #print(f"BGR pixel: {bgr_pixel}")
        # #print(f"RGB pixel: {rgb_pixel}")


        # ToDo: The loaded image should be saved in self._image.
        self._colour_type: str = colour_type


        self._image: np.ndarray = np.zeros(0)
        #print(self._image)
        self._image: np.ndarray = self._read_image
        #print(self._image)
        #print(self._image.shape)
        #print(self._image[255:, :, :])

    def get_image_data(self) -> tuple[np.ndarray, str]:
        """
        Return the image data (image and colour scheme).

        Returns:
            tuple(np.ndarray, str): Loaded image and current colour scheme.
        """
        return self._image, self._colour_type

    def show_image(self):
        """
        Show the loaded image using either matplotlib or CV2.
        """

        # ToDo: Test the CV2 and the plt options. Can you find a difference in their functionality?
        # Show the image depending on the colour type.
        if self._colour_type in ["RGB", "BGR"]:
            plt.imshow(self._image) #shows the image in the RGB format
        else:
            plt.imshow(self._image, cmap="gray") 

        plt.axis("off")
        plt.show()

        """
        cv2.imshow('Image', self._image)

        #cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

    def save_image(self, image_title: str):
        """
        Save the loaded image using either matplotlib or CV2.

        Args:
            image_title (str): Title of the image with the corresponding extension.
        """

        # Combine the image parent directory and the given title to create the path for the new image.
        total_image_path: str = os.path.join(self._image_directory, image_title)
        plt.imsave(total_image_path, self._image)

        """
        cv2.imwrite(filename=total_image_path, img=self._image)
        """

    def convert_colour(self):
        """
        Convert a colour image from BGR to RGB or vice versa.
        Do not use functions from external libraries.
        Solve this task by using indexing.
        """
        if self._colour_type not in ["RGB", "BGR"]:
            raise ValueError("The function only works for colour images!")

        # ToDo: Perform the colour conversion from RGB to BGR or vice versa.
        self._image = self._image[:, :, ::-1] #Converting Either of the images to the other swapping the R and B channels
        # cv2.imshow('Image2 -- BGR', self._image)
        # #cv2.waitKey(0)

        # ToDo: Also update the current colour scheme.

        if self._colour_type == "RGB":
            self._colour_type = "BGR"
        else:
            self._colour_type = "RGB"

        # ToDo: Do not use any external libraries or loops.

        # Perform the colour conversion.

    def clip_image(self, clip_min: int, clip_max: int):
        """
        Clip all colour values in the image to a given min and max value.
        Do not use functions from external libraries.
        Solve this task by using indexing.

        Args:
            clip_min (int): Minimum image colour intensity.
            clip_max (int): Maximum image colour intensity.
        """
        # ToDo: Clip the pixel/colour intensities of the image to predefined values.
        # ToDo: Do not use any external libraries or loops.
        
        # #print(self._image[self._image > clip_max])
        self._image[self._image < clip_min] = clip_min #take values less than Minimum and assign to Minimum
        self._image[self._image > clip_max] = clip_max #take values greater than Maximum and assign to Maximum
        # #print(self._image)



    def flip_image(self, flip_value: int):
        """
        Flip an image either vertically (0), horizontally (1) or both ways (2).
        Do not use functions from external libraries.

        Args:
            flip_value (int): Value to determine how the image should be flipped.
        """
        if flip_value not in [0, 1, 2]:
            raise ValueError("The provided flip value must be either 0, 1 or 2!")

        # ToDo: Flip the image using indexing.
        # ToDo: Do not use any external libraries or loops.
        if flip_value == 0:
            self._image = self._image[::-1, :, :] #Flipping Row values
            cv2.imshow('Image - Vertical Flip', self._image) 
            #cv2.waitKey(0)
        elif flip_value == 1:
            self._image = self._image[:, ::-1, :] #Flipping Column values
            cv2.imshow('Image - Horizontal Flip', self._image) 
            #cv2.waitKey(0)
        else:
            self._image = self._image[::-1, ::-1, :] #Flipping both Row and Column values
            cv2.imshow('Image - Both Flip', self._image) 
            #cv2.waitKey(0)


if __name__ == '__main__':
    processor = ImageProcessor(image_path=IMAGE_PATH, colour_type="RGB")
    processor.convert_colour()
    processor.clip_image(0, 255)
    processor.flip_image(2)
    processor.show_image()