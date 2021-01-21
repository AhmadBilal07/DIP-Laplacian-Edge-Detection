import cv2 as cv

# Opening Image
img = cv.imread('sample images/lena.png')

# Converting img into Gray Scale
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Applying Gaussian filter to smoothen the image i.e Noise Removal
smoothImg = cv.GaussianBlur(grayImg,(5,5),cv.BORDER_DEFAULT)

# Applying Laplacian Edge Detection Function
'''
 ddepth means destination image depth 
 CV_8U -> 8-bit unsigned integers ( 0..255 )
 ksizze means Kernal size
'''
result = cv.Laplacian(smoothImg, ddepth = cv.CV_8U, ksize=5)

# Displaying both Original and Smoothened Images
cv.imshow("Orginal Image", result)

# Waits for a key Press
cv.waitKey(0)

# destroys the window showing image
cv.destroyAllWindows()