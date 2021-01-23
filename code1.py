import numpy as np
from PIL import Image,ImageOps,ImageFilter

# Convolutional Function
def convolution(image, kernel):
    m, n = kernel.shape
    if (m == n):
        x, y = image.shape
        x = x - m + 1
        y = y - m + 1
        resultImg = np.zeros((x,y))
        for i in range(x):
            for j in range(y):
                resultImg[i][j] = np.sum(image[i:i+m, j:j+m]*kernel)
    return resultImg


if __name__ == '__main__':
    # Image Path
    IMAGE = "sample images/lena.png"

    # Opening Image using PIL
    img = Image.open(IMAGE)

    #Removing Noise & smoothing image for better performance
    img = img.filter(ImageFilter.SMOOTH)

    # Converting RGB to grayscale image
    grayImg = ImageOps.grayscale(img)

    # Converting grayscale img into array
    imgArr = np.asarray(grayImg, dtype="int32")

    # Padding Image with zeros
    imgArr = np.pad(imgArr, 1, mode='constant')

    '''
    3x3 Laplacian Mask
    '''

    kernel = np.array([
        [-1,-1,-1],
        [-1,8,-1],
        [-1,-1,-1]])

    # Applying Convolution using our kernel Function
    dstImg = convolution(imgArr, kernel)

    # Converting array to back to image
    dstImg = Image.fromarray(dstImg)

    # Displaying both original and resultant images
    grayImg.show()
    dstImg.show()