import cv2
from transform import hough_windowed_rectangle, hough_cricles
from utils import plot_hough_space, graph_output, graph_rectangles_and_circles, countMoney
from skimage.io import imshow
import matplotlib.pyplot as plt

img = cv2.imread("cases/case1.png")
edge_image, img_lines, img_rectangles, hough_space, peaks, lines, rectangles, areas = hough_windowed_rectangle(img)
img_w_circles, circles = hough_cricles(img, region=20, threshold=15, r_range=(40, 60))
print("Money Count in Pounds", countMoney(circles, areas))
img_w_rect_circles = graph_rectangles_and_circles(img, rectangles, circles)
graph_output(edge_image, img_lines, img_w_rect_circles, hough_space, peaks)
