import numpy as np

def make_color_image(image_list, color_list = ['gray', 'cyan', 'yellow', 'magenta', 'blue', 'green', 'red']):
    """
    Numpy based function to merge multiple images into one color image. 

    Put in a list of images you want to merge, followed by the list of corresponding colors (in correct order).
    If you want to add a new color, add this to the color_info and set the multiplier of each channel.
    Example for orange is listed and commented out. 

    Limitations:
    - Need to use > 1 image (prety logical right?)
    - Not possible to use rgb images at the moment (only graytone) 
        This can be changed if needed, but will take some more time
    - Images need the same shape
    """
    
    # make the color_info dictionary
    color_info = {}
    color_info['blue'] = (0.0, 0.0, 1.0)
    color_info['green'] = (0.0, 1.0, 0.0)
    color_info['red'] = (1.0, 0.0, 0.0)
    color_info['yellow'] = (1.0, 1.0, 0.0)
    color_info['cyan'] = (0.0, 1.0, 1.0)
    color_info['magenta'] = (1.0, 0.0, 1.0)
    color_info['gray'] = (1.0, 1.0, 1.0)
    # color_info['orange'] = (0.9, 0.5, 0.0)
    
    # check if we have enough images
    if len(image_list) < 2:
        print('Please use a list of >1 images as image_list input', flush = True)
        return None
    # check if we have enough colors
    if len(color_list) < len(image_list):
        print('Please use the same amount of colors as images.', flush = True)
        return None
    # check if all images have the same shape
    shapes = set()
    for image in image_list:
        shapes.add(image.shape)
    if len(shapes) > 1:
        print('All input images must have the same shape / size.', flush = True)
        return None

    # create empty images
    red_image = np.zeros_like(image_list[0])
    green_image = np.zeros_like(image_list[0])
    blue_image = np.zeros_like(image_list[0])

    # put the image in the correct rgb stack
    for image, color in zip(image_list, color_list):
        rgb_info = color_info[color]
        if len(image.shape) > 2:
            image = np.max(image, axis = 2)
        if rgb_info[0] > 0.0:
            red_image = np.dstack((red_image, (rgb_info[0] * image).astype(image.dtype)))
        if rgb_info[1] > 0.0:
            green_image = np.dstack((green_image, (rgb_info[1] * image).astype(image.dtype)))
        if rgb_info[2] > 0.0:
            blue_image = np.dstack((blue_image, (rgb_info[2] * image).astype(image.dtype)))

    # make the maximum projection of each color stack
    if len(red_image.shape) > 2:
        red_image = np.max(red_image, axis = 2)
    if len(green_image.shape) > 2:
        green_image = np.max(green_image, axis = 2)
    if len(blue_image.shape) > 2:
        blue_image = np.max(blue_image, axis = 2)

    # get the color image
    color_image = np.dstack((red_image, green_image, blue_image))

    return color_image

# define functions
def convert_image(image, new_type = np.uint8, new_minimum = False, new_maximum = False, min_intensity_q = False, max_intensity_q = False):
    """
    Numpy based function to convert any image. 

    Has multiple uses:
        - Convert to a different data type (eg. from np.uint8 to np.uint16)
        - Cropping / stretching the intensity range by using new_minimum and new_maximum
        - Background removal using the minimal_image_q(uantile)
        - Enhancing low intensities by either the new_maximum or maximum_image_q(uantile) 
    
    Binary images can be used, but will only be useful to change data type.
    """
    # get numpy type info about the mimimum and maximum values
    type_info = {}
    type_info[np.uint8] = {'minimum':0, 'maximum':255}
    type_info[np.uint16] = {'minimum':0, 'maximum':65535}
    type_info[np.uint32] = {'minimum':0, 'maximum':4294967295}
    type_info[np.uint64] = {'minimum':0, 'maximum':18446744073709551615}
    type_info[np.int8] = {'minimum':-128, 'maximum':127}
    type_info[np.int16] = {'minimum':-32768, 'maximum':32767}
    type_info[np.int32] = {'minimum':-2147483648, 'maximum':2147483647}
    type_info[np.int64] = {'minimum':-9223372036854775808, 'maximum':9223372036854775807}  
    
    # set the minimum and maximum values if they are not provided
    if new_minimum == False:
        new_minimum = type_info[new_type]['minimum']
    if new_maximum == False:
        new_maximum = type_info[new_type]['maximum']

    #Get the minimum value to use to stretch the intensity range
    image_minimum = image.min()
    #Correct binary image
    if image_minimum == False:
        image = image * 1
        image_minimum = image.min()
    #Overwrite the image_minimum by the quantile if this is defined
    if min_intensity_q != False:
        image_minimum = np.quantile(image, q = min_intensity_q)
        
    #Get the maximum value to use to stretch the intensity range
    image_maximum = image.max()
    #Overwrite the image_maximum by the quantile if this is defined
    if max_intensity_q != False:
        image_maximum = np.quantile(image, q = max_intensity_q)
    
    #Build the new image
    a = (new_maximum - new_minimum)/(image_maximum - image_minimum)
    b = new_maximum - a * image_maximum
    new_image = (a * image + b)

    new_image[new_image > new_maximum] = new_maximum
    new_image[new_image < new_minimum] = new_minimum
    new_image = new_image.astype(new_type)
    return new_image