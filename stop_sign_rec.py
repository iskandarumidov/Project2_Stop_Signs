import numpy as np
import math
import copy
import sys
from PIL import Image
import os

def read_image():
    # Read the image from hard drive
    input_image = open(os.path.join(os.getcwd(), "Input Images/image-5.bmp"), "rb")

    # Skip two bytes. Then read the following 4 bytes in little endian to get size of image
    input_image.seek(2)
    print("Image size in bytes: ", int.from_bytes(input_image.read(4), byteorder='little'))

    # Read height and width at offset 18. Each is 4 bytes long
    input_image.seek(18, 0)
    image_width = int.from_bytes(input_image.read(4), byteorder='little')
    image_height = int.from_bytes(input_image.read(4), byteorder='little')
    print("Image dimensions, width * height: ", image_width, "*", image_height)

    # Read # of bits per pixel starting at offset 28
    input_image.seek(28, 0)
    bytes_per_pixel = int.from_bytes(input_image.read(2), byteorder='little') // 8  
    print("Bytes per pixel is: ", bytes_per_pixel)

    # Now we are getting to actual content. Read pic into a Numpy array
    input_image.seek(54, 0)  
    data_in_pixel_all_row = None
    data_in_pixel_all_row = np.array([[[each_byte for each_byte in reversed(input_image.read(bytes_per_pixel))]
                                       for i in range(image_width)] for j in range(image_height)], np.uint8)

    print("shape of the numpy array is: ", data_in_pixel_all_row.shape)
    # Return representation of image in Numpy
    return data_in_pixel_all_row


def hough_transform():
    # Get the thinned image first
    thinned_edge_image = compare()
    # Threshold value above which the pixel values will be considered part of line
    line_threshold = 150
    local_wd = 20
    theta_max = math.pi
    theta_min = 0.0
    d_min = 0.0
    d_max = math.hypot(thinned_edge_image.shape[0], thinned_edge_image.shape[1])
    d_threshold = 200
    theta_threshold = 300
    accumulator_array = np.zeros((d_threshold, theta_threshold))

    for x in range(thinned_edge_image.shape[0]):
        for y in range(thinned_edge_image.shape[1]):

            if thinned_edge_image[x, y] == 255:
                continue
            for i_theta in range(theta_threshold):
                theta = i_theta * theta_max / theta_threshold
                d = x * math.cos(theta) + y * math.sin(theta)

                i_d = int(d_threshold * d / d_max)
                accumulator_array[i_d, i_theta] = accumulator_array[i_d, i_theta] + 1

    # Return image with lines
    return accumulator_array


def perform_formatting():
    image_3d_array = read_image()
    edge_transform_array_x = np.zeros(shape=(image_3d_array.shape[0], image_3d_array.shape[1]))
    edge_transform_array_y = np.zeros(shape=(image_3d_array.shape[0], image_3d_array.shape[1]))
    # Sobel
    lapx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    lapy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    img_height = image_3d_array.shape[0]
    img_width = image_3d_array.shape[1]

    # Brightness + contrast
    for i in range(img_height):    # For every pixel:
        for j in range(img_width):
            image_3d_array[i, j] = [127 + (((rbg + 20) - 127) * 1.0) for rbg in image_3d_array[i, j]]
    test_driver_image(image_3d_array)
    histogram_buffer = np.zeros((256, 3), dtype=np.int32)  # Initialize histogram buffer to zero

    # Histogram
    for i in range(img_height):
        for j in range(img_width):
            for pix in range(histogram_buffer.shape[1]):
                histogram_buffer[image_3d_array[i, j, pix], pix] += 1

    # Cumulative histogram
    for i in range(1, histogram_buffer.shape[0]):
        for j in range(0, histogram_buffer.shape[1]):
            histogram_buffer[i, j] = histogram_buffer[i - 1, j] + histogram_buffer[i, j]

    # Normalize cumulative histogram
    for i in range(histogram_buffer.shape[0]):
        for j in range(histogram_buffer.shape[1]):
            histogram_buffer[i, j] = histogram_buffer[i, j] * 255 / (img_height * img_width)
    for i in range(img_height):
        for j in range(img_width):
            image_3d_array[i, j] = [histogram_buffer[image_3d_array[i, j, pix], pix] for pix in range(histogram_buffer.shape[1])]

    image_2d_array = convert_2d_3d(image_3d_array)

    print(image_2d_array.shape)
    edge_transform_array_x = np.zeros((image_2d_array.shape[0] + 2, image_2d_array.shape[1] + 2))
    edge_transform_array_y = np.zeros((image_2d_array.shape[0] + 2, image_2d_array.shape[1] + 2))

    # Laplacian, 3x3

    for i in range(2, edge_transform_array_x.shape[0] - 2):
        for j in range(2, edge_transform_array_x.shape[1] -2):

            edge_transform_array_x[i, j] = sum(lapx[m, n] * image_2d_array[i - m,  j - n] for m in range(3) for n in range(3))

    for i in range(2, edge_transform_array_y.shape[0] - 2):
        for j in range(2, edge_transform_array_y.shape[1]- 2):

            edge_transform_array_y[i, j] = sum(lapy[m, n] * image_2d_array[i - m,  j - n]
                                               for m in range(3) for n in range(3))

    dup_eta = np.abs(np.subtract(edge_transform_array_x, edge_transform_array_y))
    # Remove the padded zeros
    dup_eta = dup_eta[2:-2, 2:-2]           
    for i in range(dup_eta.shape[0]):
        for j in range(dup_eta.shape[1]):
            if dup_eta[i, j] > 100:
                dup_eta[i, j] = 0

            else:
                dup_eta[i, j] = 255
    test_driver_image(dup_eta)
    ####################################### Reducing the noise from the edge image ##################################
    # If number of neighbouring element is greater than, certain threshold then keep edge element
    # else remove it. A better approximation of threshold can be 2 or 3

    threshold_noise = 8  # Total number of neighbouring elements for a pixel to not being identified as noise

    for i in range(2, dup_eta.shape[0] - 2):
        for j in range(2, dup_eta.shape[1] - 2):
            # check if the pixel element in black i.e., it constitutes an edge
            if dup_eta[i, j] != 255:
                # Check neighbouring pixel value in 5*5 window, if set to 0 i.e., black, increase the count
                cur_threshold = sum(1 for m in range(-2, 3) for n in range(-2, 3) if dup_eta[i + m, j + n] == 0)

                if cur_threshold - 1 < threshold_noise:

                    dup_eta[i, j] = 255

    test_driver_image(dup_eta)
    # Return reformatted image, after smoothing, contrast management and other formatting operations
    return dup_eta


import reformat_image
import numpy as np
from test_image import test_driver_image
from reformat_image import perform_formatting


def compare():
    edge_image = perform_formatting()
    # We use two images -> 1 . original image pixels
    # 2nd -> final image pixels
    # we loop until original image pixels and final image pixels are not identical
    # Contour points ->  Edge pixel that is bounded in at least one side by non edge element, that is removed
    # create a final image of size corresponding to edge image, initialized with all white pixel value i.e, 255
    final_image = np.full((edge_image.shape[0], edge_image.shape[1]), 255)
    # We maintain 2 set of points Ai and Bi to determine the final points
    # None is the placeholder for the values x and y, which are conditional
    # np.nan is the placeholder for the value, for which the values at corresponding location is not defined
    # Ai are always final and Bi are final depending upon the sub-cycles.

    Ai = []
    Bi = []

    A1 = np.array([[None, None, None], [255, 0, 255], [None, None, None]])
    A2 = np.array([[255, None, None], [None, 0, None], [None, None, 255]])
    A3 = np.array([[None, 255, None], [None, 0, None], [None, 255, None]])
    A4 = np.array([[None, None, 255], [None, 0, None], [255, None, None]])

    Ai.extend([A1, A2, A3, A4])

    B1 = np.array([[None, None, None], [np.NaN, 0, 255], [255, 0, np.NaN]])
    B2 = np.array([[None, np.nan, 255], [None, 0, 0], [None, 255, np.nan ]])
    B3 = np.array([[np.nan, 0, 255], [255, 0, np.nan], [None, None, None]])
    B4 = np.array([[np.nan, 255, None], [0, 0, None], [255, np.nan, None]])

    Bi.extend([B1, B2, B3, B4])
    b_mappings = {0: (1, 2), 1: (3, 4), 2: (1, 4), 3: (2, 3)}
    while edge_image.all() != final_image.all():
        # Each sub-cycle iterates 4 times, to prevent outcome to be non-connected lines while transformation
        # The sub- cycles are pixel with (non-edge element)
        #                                   -> South border set to 255
        #                                   -> North border set to 255
        #                                   -> West border set to 255
        #                                   -> East border set to 255
        for q in range(4):
            ontour_points = np.full((edge_image.shape[0], edge_image.shape[1]), 255)
            # First extract the final points
            for i in range(1, edge_image.shape[0] - 1):
                for j in range(1, edge_image.shape[1] - 1):
                    # This creates a 3*3 test window, by extracting values from the array
                    test_window = np.array([[edge_image[i + m, j + n] for n in range(-1, 2)] for m in range(-1, 2)])
                    outer_res = False
                    # Compare the above test window with Ai adn Bi to detect tge final points
                    for a_window in Ai:
                        res = True
                        for a_window_i in range(a_window.shape[0]):
                            for a_window_j in range(a_window.shape[1]):

                                if a_window[a_window_i, a_window_j] is not None:
                                    if a_window[a_window_i, a_window_j] != test_window [a_window_i, a_window_j]:
                                        res = False
                        if res:  # If after comparing, it is still true, then it's a final point
                            outer_res = True
                            break
                    if outer_res:
                        final_image [i, j] = 0      # Set this index (i,j) in the final element matrix
                        continue
                    else:
                        bs = b_mappings[q]
                        for ind_b in bs:
                            b_window = Bi[ind_b - 1]
                            res = True
                            for b_window_i in range(b_window.shape[0]):
                                for b_window_j in range(b_window.shape[1]):

                                    if b_window[b_window_i, b_window_j] is not None:
                                        if b_window[b_window_i, b_window_j] is not np.nan:
                                            if b_window[b_window_i, b_window_j] != test_window[b_window_i, b_window_j]:
                                                res = False
                            if res:  # If after comparing, it is still true, then it's a final point

                                final_image[i, j] = 0 # Set to black, if this pixel matches any of B's
                                break
            # By this point, it is determined that if any pixels in the whole image array belongs to final point or not
            for i in range(1, edge_image.shape[0] - 1):
                for j in range(1, edge_image.shape[1] - 1):

                    if q == 0:

                        if edge_image[i + 1, j] == 255:

                            edge_image[i, j] = 255  # Remove the contour points
                    elif q == 1:
                        if edge_image[i - 1, j] == 255:
                            edge_image[i, j] = 255
                    elif q == 2:

                        if edge_image[i, j-1] == 255:
                            edge_image[i, j] = 255
                    else:
                        if edge_image[i, j + 1] == 255:
                            edge_image[i, j] = 255
            # After all the contour points is deleted from the original image, add the final points to final image
            for i in range(1, edge_image.shape[0] - 1):
                for j in range(1, edge_image.shape[1] - 1):
                    # Add the final points in the image
                    if final_image[i, j] == 0:
                        edge_image[i, j] = 0

    test_driver_image(final_image)

    return final_image


def convert_2d_3d(input_array):
    dimension_converted_array = None

    if input_array.ndim == 3:
        # Convert into 2 dimensional
        dimension_converted_array = np.zeros((input_array.shape[0], input_array.shape[1]), dtype=np.uint8)
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):
                dimension_converted_array[i, j] = input_array[i, j, 1]

    else:
        # Convert into 3 dimensional
        dimension_converted_array = np.zeros((input_array.shape[0], input_array.shape[1], 3), dtype=np.uint8)
        for i in range(input_array.shape[0]):
            for j in range(input_array.shape[1]):

                dimension_converted_array[i, j] = [input_array[i, j] for p in range(3)]

    return dimension_converted_array



def test_driver_image(ndarray):


    ndarray = ndarray if ndarray.ndim == 3 else convert_2d_3d(ndarray)

    # Since bitmap images are read in bottom-up fashion, reverse the ndarray
    image_2d_array = ndarray[::-1, :, :]

    img = Image.new( 'RGB', (image_2d_array.shape[1], image_2d_array.shape[0]), "black") # Create a new black image
    pixels = img.load() # Create the pixel map
    for i in range(img.size[0]):    # For every pixel:
        for j in range(img.size[1]):
            pixels[i, j] = tuple(image_2d_array[j, i])  # Set the colour accordingly


    img.show()

    img.save(os.path.join(os.getcwd(),"teste_image.bmp"))
