import numpy as np
from PIL import Image

############### ---------- Basic Image Processing ------ ##############

### TODO 1: Read an Image and convert it into a floating point array with values between 0 and 1. You can assume a color image
def imread(filename):
    img = np.array(Image.open(filename).convert('RGB'))
    img = np.float32(img) / 255.    
    return img


### TODO 2: Convolve an image (m x n x 3 or m x n) with a filter(l x k). Perform "same" filtering. Apply the filter to each channel if there are more than 1 channels
def convolve(img, filt):

    kernel = np.flipud(np.fliplr(filt))

    if(img.ndim == 2):
        convolved = convolveHelper(img, kernel)
        return convolved
    else:
        channels = img.shape[2]
        final_img = np.zeros(img.shape)
        for i in range(channels): 
            convolved = convolveHelper(img[:,:,i], kernel)
            final_img[:,:,i] = convolved
        return final_img


### convolve each channel
def convolveHelper(img, kernel): 

    #img_pad = np.pad(img, 1, mode='constant')
    output = np.zeros(shape = (img.shape[0], img.shape[1]))
    # print("kernel")
    # print(kernel.shape[0])
    # print(kernel.shape[1])

    for i in range(0, img.shape[1]):
        for j in range(0, img.shape[0]): 
            sum = 0
            col = 0
            for x in range(i - (kernel.shape[1]//2), i + (kernel.shape[1]//2) + 1):
                row = 0
                for y in range(j - (kernel.shape[0]//2), j + (kernel.shape[0]//2) + 1):
                    if((0 <= x < img.shape[1]) and (0 <= y < img.shape[0])):
                        sum += kernel[row, col] * img[y, x]
                    row += 1
                col += 1
            output[j, i] = sum

    return output


### TODO 3: Create a gaussian filter of size k x k and with standard deviation sigma
def gaussian_filter(k, sigma):
    ax = np.linspace(-(k - 1) / 2.0, (k - 1) / 2.0, k)
    xx, yy = np.meshgrid(ax, ax)
    goose = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
    return goose / np.sum(goose)


### TODO 4: Compute the image gradient. 
### First convert the image to grayscale by using the formula:
### Intensity = Y = 0.2125 R + 0.7154 G + 0.0721 B
### Then convolve with a 5x5 Gaussian with standard deviation 1 to smooth out noise. 
### Convolve with [0.5, 0, -0.5] to get the X derivative on each channel
### convolve with [[0.5],[0],[-0.5]] to get the Y derivative on each channel
### Return the gradient magnitude and the gradient orientation (use arctan2)
def gradient(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    img_gray = R*0.2125 + G*0.7154 + B*0.0721
    gaussian = gaussian_filter(5, 1)
    x_filt = np.array([[0.5, 0, -0.5]])
    y_filt = np.array([[0.5],[0],[-0.5]])
    img_convolve = convolve(img_gray, gaussian)
    x_der = convolve(img_convolve, x_filt)
    y_der = convolve(img_convolve, y_filt)
    mag = np.hypot(x_der, y_der)
    ori = np.arctan2(y_der, x_der)
    return mag, ori


##########----------------Line detection----------------

### TODO 5: Write a function to check the distance of a set of pixels from a line parametrized by theta and c. The equation of the line is:
### x cos(theta) + y sin(theta) + c = 0
### The input x and y are numpy arrays of the same shape, representing the x and y coordinates of each pixel
### Return a boolean array that indicates True for pixels whose distance is less than the threshold
def check_distance_from_line(x, y, theta, c, thresh):
    # Calculate the distance from each point (x, y) to the line

    dist = np.abs(x*np.cos(theta) + y*np.sin(theta) + c)

    # Return a boolean array indicating which points are within the threshold
    return dist < thresh


### TODO 6: Write a function to draw a set of lines on the image. 
### The `img` input is a numpy array of shape (m x n x 3).
### The `lines` input is a list of (theta, c) pairs. 
### Mark the pixels that are less than `thresh` units away from the line with red color,
### and return a copy of the `img` with lines.
def draw_lines(img, lines, thresh):

    y = np.arange(img.shape[1])
    x = np.arange(img.shape[0])

    img_copy = np.copy(img)
    for j in range(img_copy.shape[0]):
        x = np.array([j]*len(img_copy[0]))
        for (theta, c) in lines: 
            dist = check_distance_from_line(y, x, theta, c, thresh)
            for val in range(x.shape[0]):
                if(dist[val] == True):
                    img_copy[x[val]][y[val]][0] = 1
                    img_copy[x[val]][y[val]][1] = 0
                    img_copy[x[val]][y[val]][2] = 0
    return img_copy


### TODO 7: Do Hough voting. You get as input the gradient magnitude (m x n) and the gradient orientation (m x n), 
### as well as a set of possible theta values and a set of possible c values. 
### If there are T entries in thetas and C entries in cs, the output should be a T x C array. 
### Each pixel in the image should vote for (theta, c) if:
### (a) Its gradient magnitude is greater than thresh1, **and** 
### (b) Its distance from the (theta, c) line is less than thresh2, **and**
### (c) The difference between theta and the pixel's gradient orientation is less than thresh3
def hough_voting(gradmag, gradori, thetas, cs, thresh1, thresh2, thresh3):
    threshold_two = []
    threshold_one = []

    votes = np.zeros((thetas.shape[0], cs.shape[0]))
    threshold_one = np.where(gradmag > thresh1)

    for theta in range(thetas.shape[0]):
        for c in range(cs.shape[0]): 
            dist = check_distance_from_line(threshold_one[1], threshold_one[0], thetas[theta], cs[c], thresh2)
            threshold_two = np.where(dist)[0]
            for pixel in threshold_two:
                if np.abs(thetas[theta] - gradori[threshold_one[0][pixel], threshold_one[1][pixel]]) < thresh3:
                    votes[theta][c] += 1
    return votes

    
### TODO 8: Find local maxima in the array of votes. A (theta, c) pair counts as a local maxima if: 
### (a) Its votes are greater than thresh, **and** 
### (b) Its value is the maximum in a nbhd x nbhd beighborhood in the votes array.
### The input `nbhd` is an odd integer, and the nbhd x nbhd neighborhood is defined with the 
### coordinate of the potential local maxima placing at the center.
### Return a list of (theta, c) pairs.
def localmax(votes, thetas, cs, thresh, nbhd):
    local_max = []
    for theta in range(thetas.shape[0]):
        for c in range(cs.shape[0]): 
            if votes[theta][c]>thresh:
                max = votes[theta][c]
                for x in range(theta - (nbhd//2), theta + (nbhd//2) + 1):
                    for y in range(c - (nbhd//2), c + (nbhd//2) + 1):
                        if((0 <= x < votes.shape[1]) and (0 <= y < votes.shape[0])):
                            if(votes[x][y] >= max):
                                max = votes[x][y]
                if(max == votes[theta][c]):
                    local_max.append((thetas[theta],cs[c]))
    return local_max
                

# Final product: Identify lines using the Hough transform    
def do_hough_lines(filename):

    # Read image in
    img = imread(filename)

    # Compute gradient
    gradmag, gradori = gradient(img)

    # Possible theta and c values
    thetas = np.arange(-np.pi-np.pi/40, np.pi+np.pi/40, np.pi/40)
    imgdiagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
    cs = np.arange(-imgdiagonal, imgdiagonal, 0.5)

    # Perform Hough voting
    votes = hough_voting(gradmag, gradori, thetas, cs, 0.1, 0.5, np.pi/40)

    # Identify local maxima to get lines
    lines = localmax(votes, thetas, cs, 20, 11)

    # Visualize: draw lines on image
    result_img = draw_lines(img, lines, 0.5)

    # Return visualization and lines
    return result_img, lines



##FINAL
   
    
