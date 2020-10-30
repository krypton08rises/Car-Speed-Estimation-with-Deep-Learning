import cv2
import numpy as np
import matplotlib.pyplot as plt


def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return  # Make a copy of the original image.
    img = np.copy(img)  # Create a blank image that matches the original in size.
    # Loop over all lines and draw them on the blank image.
    print(lines.shape)
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # for i, line in enumerate(lines):
    for x1, y1, x2, y2 in lines[9]:
            # Merge the image with the lines onto the original.
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    for x1, y1, x2, y2 in lines[15]:
            # Merge the image with the lines onto the original.
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)  # Return the modified image
        
    # plt.figure()
    # plt.imshow(line_image)
    # plt.show()
    # cv2.imshow('line', line_image)
    # cv2.waitKey(0)

    # print(img.shape, line_image.shape)





# Read and display Image (Change path as per your computer)
def get_lines():

    img = cv2.imread('./frames/frame168.jpg')
    img = cv2.resize(img, (416, 416))
    temp = img
    # ing = cv2.imread('/Users/Sumanth/Desktop/GIS/edge.tiff')

    triangle_1 = np.array([[0, 0], [0, 365], [390, 0]])
    triangle_2 = np.array([[416, 416], [325, 416], [416, 100]])
    color = [0, 0, 0]
    cv2.fillConvexPoly(img, triangle_1, color)
    cv2.fillConvexPoly(img, triangle_2, color)
    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 10, 50, 255)

    # cv2.imshow('edges', edged)
    # cv2.waitKey(0)

    lines = cv2.HoughLinesP(edged,
                            rho=6,
                            theta=np.pi / 60,
                            threshold=160,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=25)
    print(lines[9])
    print(lines[15])
    draw_lines(temp, lines)

    return lines[9], lines[15]


# get_lines()/
# cv2.imwrite('sample.jpg', img)
# cv2.imshow('image', img)
# cv2.waitKey(0)
