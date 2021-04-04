from skimage.draw import line
import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
import numpy as np
import matplotlib.pyplot as plt

def get_lines_intersection(line1, line2, WIDTH, HEIGHT):
    rho_1, a_1, b_1 = line1[0], np.cos(np.deg2rad(line1[1])), np.sin(np.deg2rad(line1[1]))
    x0, y0 = (a_1 * rho_1) + WIDTH / 2, (b_1 * rho_1) + HEIGHT / 2
    x1, y1 = int(x0 + WIDTH * (-b_1)), int(y0 + WIDTH * (a_1))
    x2, y2 = int(x0 - WIDTH * (-b_1)), int(y0 - WIDTH * (a_1))

    rho_2, a_2, b_2 = line2[0], np.cos(np.deg2rad(line2[1])), np.sin(np.deg2rad(line2[1]))
    x02, y02 = (a_2 * rho_2) + WIDTH / 2, (b_2 * rho_2) + HEIGHT / 2
    x3, y3 = int(x02 + WIDTH * (-b_2)), int(y02 + WIDTH * (a_2))
    x4, y4 = int(x02 - WIDTH * (-b_2)), int(y02 - WIDTH * (a_2))

    D = (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4)
    p_x = int(((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/D)
    p_y = int(((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/D)
    return [p_x, p_y]

def hough_windowed_rectangle(img, num_rhos=None, num_thetas=None, threshold=None):
    """[summary]

    Args:
        img ([numpy array]): [the original BGR image]
        num_rhos ([int], optional):
            Specifies the number of rhos in the hough space
            which determines the delta_rho, which is the smallest
            increment between two conseuctive rhos. The larger
            num_rhos the more resolution the hough space will have of rhos

            Defaults to None.
        num_thetas ([int], optional): 
            Specifies the number of thetas in the hough space
            which determines the delta_theta, which is the smallest
            increment between two conseuctive thetas. The larger
            num_thetas the more resolution the hough space will have of thetas
            Defaults to None.
        threshold ([int], optional): 
            The value of the local maxima in the hough space from which it can be
            inferred that a line is detected. The higher the less sensitive
            the detection is.
            Defaults to None.

    Returns:
        [type]: [description]
    """
    img_ = cv2.medianBlur(img, 7)
    img_ = cv2.GaussianBlur(img_, (7, 7), 0)
    img_ = cv2.bilateralFilter(img_, 9, 75, 75)
    edge_image = cv2.Canny(img, 160, 1100)
    WIDTH, HEIGHT = edge_image.shape[0], edge_image.shape[1]

    if num_rhos is None:
        num_rhos = int((WIDTH + HEIGHT)/2)
    if num_thetas is None:
        num_thetas = int((WIDTH + HEIGHT)/2)
    if threshold is None:
        threshold = int((WIDTH + HEIGHT)*0.157)
    
    # get diameter of the image = longest line to exist
    d = np.sqrt(WIDTH**2 + HEIGHT**2)

    delta_th = 180/num_thetas
    delta_rho = (2*d)/num_rhos

    thetas = np.arange(0, 180, step=delta_th)
    rhos = np.arange(-d, d, step=delta_rho)

    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))

    edges_x_y = np.argwhere(edge_image != 0)
    # shift x, y to be from -d to d
    edges_x_y = edges_x_y - np.array([[HEIGHT/2, WIDTH/2]])

    #calculate valid rho values given edges x,y and the thetas domain
    rho_values = np.matmul(edges_x_y, np.array([sin_thetas, cos_thetas]))

    # build accumulator space using histogram2d
    accumulator, theta_vals, rho_vals = np.histogram2d(
        np.tile(thetas, rho_values.shape[0]),
        rho_values.ravel(),
        bins=[thetas, rhos]
    )

    accumulator = np.transpose(accumulator)
    lines = np.argwhere(accumulator > threshold)
    
    
    # set similarity threshold to lines so that only unique lines are kept
    lines_similarity_threshold = 0.05*(WIDTH+HEIGHT)/2
    # construct unique lines array
    unique_lines = []
    first = True
    for line in lines:
        if not first:
            th_prev = thetas[prev_line[1]] % 180
            th_curr = thetas[line[1]] % 180
            difference = np.abs(rhos[prev_line[0]] - rhos[line[0]])\
                +np.abs(th_curr - th_prev)
            if np.sum(difference) > lines_similarity_threshold:
                unique_lines.append(line)
        prev_line = line
        if first:
            unique_lines.append(line)
            first = False
    unique_lines = np.array(unique_lines)
    rho_idxs, theta_idxs = unique_lines[:, 0], unique_lines[:, 1]
    H_rhos, H_thetas = rhos[rho_idxs], thetas[theta_idxs]
    # every line is (rho, theta)
    # draw detected unique lines
    detected_lines_st_end_pts = []
    image_with_lines = img.copy()
    for line in unique_lines:  
        y, x = line
        rho = rhos[y]
        theta = thetas[x]
        a = np.cos(np.deg2rad(theta))
        b = np.sin(np.deg2rad(theta))
        x0 = (a * rho) + WIDTH / 2
        y0 = (b * rho) + HEIGHT / 2
        x1 = int(x0 + WIDTH * (-b))
        y1 = int(y0 + WIDTH * (a))
        x2 = int(x0 - WIDTH * (-b))
        y2 = int(y0 - WIDTH * (a))
        detected_lines_st_end_pts.append([[x1, x2], [y1, y2]])
        image_with_lines = cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # to detect rectangles, we need to find all pairs of parallel lines
    # then match parallel pair that are perpendicular to each other
    EPS_theta, EPS_alpha = 3, 3
    extended_peaks_P = []
    parallel_lines = []
    for i in range(len(unique_lines)-1):
        for j in np.arange(i+1, len(unique_lines)):
            rho_i, theta_i = rhos[unique_lines[i][0]], thetas[unique_lines[i][1]]
            rho_j, theta_j = rhos[unique_lines[j][0]], thetas[unique_lines[j][1]]
            delta_th = np.abs(theta_i - theta_j)
            delta_rho = np.abs(rho_i + rho_j)
            alpha = 0.5*(theta_i+theta_j)
            zeta = 0.5*(rho_i - rho_j)
            if delta_th < EPS_theta:
                extended_peaks_P.append((alpha, zeta))
                parallel_lines.append(([rho_i, theta_i], [rho_j, theta_j]))

    rectangles = []       
    for i in range(len(extended_peaks_P)-1):
        for j in np.arange(i+1, len(extended_peaks_P)):
            alpha_k, alpha_l = extended_peaks_P[i][0],  extended_peaks_P[j][0]

            if np.abs(np.abs(alpha_k-alpha_l)-90) < EPS_alpha:
                rectangles.append((parallel_lines[i], parallel_lines[j]))
        
      
    # get points of a rectangle
    area_threshold = 140000
    rectangles_boxes = []
    count = 1
    img_w_rectangles = img.copy()
    areas = []
    for rectangle in rectangles:
        # we want the intersections of the perpidecular lines only
        P1 = get_lines_intersection(rectangle[0][0],rectangle[1][0], WIDTH, HEIGHT)
        P2 = get_lines_intersection(rectangle[0][0],rectangle[1][1], WIDTH, HEIGHT)
        P3 = get_lines_intersection(rectangle[0][1],rectangle[1][0], WIDTH, HEIGHT)
        P4 = get_lines_intersection(rectangle[0][1],rectangle[1][1], WIDTH, HEIGHT)
        points = np.array([P1, P2, P3, P4])
        rect = cv2.minAreaRect(points)
        (width, height) = rect[1]
        area = width*height
        # convert rect to 4 points format
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #area_calc = np.abs( (P1[0]*P2[1]-P1[1]*P2[0]) + P2[0]*P3[1]-P2[1]*P3[0]+ P3[0]*P4[1]-P3[1]*P4[0])/2
        if area > area_threshold:
            img_w_rectangles = cv2.drawContours(img_w_rectangles, [box], 0, (0, 0, 255), 5)
            rectangles_boxes.append([box])
            print("Detected Rectangle", count, "width=", int(width), "height=", int(height))
            print("Corner points=\n", box)
            count +=1
            areas.append((width, height))
    print("Overall detected rectangles count = ", count-1)
    return edge_image, image_with_lines, img_w_rectangles, (rho_values, thetas),\
         (H_thetas, H_rhos), detected_lines_st_end_pts, rectangles_boxes, areas

def hough_cricles(img, threshold=None, region=None, r_range=(10, 60)):
    img_ = cv2.GaussianBlur(img, (5, 5), 0)
    img_ = cv2.medianBlur(img_, 7)
    img_ = cv2.bilateralFilter(img_, 9, 75, 75)
    edge_image = cv2.Canny(img_, 100, 400)
    WIDTH, HEIGHT = edge_image.shape[0], edge_image.shape[1]

    threshold = 15
    region = 20

    if r_range is None:
        r_max = np.max(WIDTH, HEIGHT)
        r_min = 2
    else:
        r_min = r_range[0]
        r_max = r_range[1]

    padding = 2*r_max

    a = np.zeros((r_max, WIDTH + padding, HEIGHT + padding))
    b = np.zeros((r_max, WIDTH + padding, HEIGHT + padding))

    thetas = np.deg2rad(np.arange(0, 360))
    cos_thetas = np.cos(thetas)
    sin_thetas = np.sin(thetas)

    x_y_edges = np.argwhere(edge_image != 0)

    for r in np.arange(r_min, r_max, step=1):
        d = 2*(r+1)
        circle_region = np.zeros((d, d))
        for cos_theta, sin_theta in zip(cos_thetas, sin_thetas):
            x = int(np.round(r*cos_theta))
            y = int(np.round(r*sin_theta))
            circle_region[x + (r+1), y + (r+1)] = 1
        c = np.argwhere(circle_region).shape[0]
        for edge in x_y_edges:
            x = edge[0]
            y = edge[1]

            X = [x - (r+1) + r_max, x + (r+1) + r_max]
            Y = [y - (r+1) + r_max, y + (r+1) + r_max]

            a[r, X[0]:X[1], Y[0]:Y[1]] += circle_region
        a[r][a[r]<threshold*c/r] = 0

    for el in np.argwhere(a):
        r, x, y = el[0], el[1], el[2]
        temp = a[r-region:r+region, x-region:x+region, y-region:y+region]
        try:
            p, a_, b_ = np.unravel_index(np.argmax(temp), temp.shape)
        except:
            continue
        b[r+(p-region), x+(a_-region),y+(b_-region)] = 1
    circles = np.argwhere(b[:, r_max:-r_max, r_max:-r_max])

    img_w_circles = img.copy()
    count = 1
    for r,x,y in circles:
        img_w_circles = cv2.circle(img_w_circles, (y, x), r, (0, 0, 255), 4)
        print("Detected Circle", count, "r=", r, "cetner=(", y, ",", x, ")")
        count+=1
    print("Overall detected circles count = ", count-1)
    return img_w_circles, circles


