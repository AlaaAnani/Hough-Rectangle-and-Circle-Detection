
import cv2
import matplotlib.pyplot as plt
def plot_hough_space(sp, hough_space, peaks):
    thetas = hough_space[1]
    rho_values = hough_space[0]
    (H_thetas, H_rhos) = peaks

    for rho in rho_values:
        sp.plot(thetas, rho, color="white", alpha=0.05) 
    sp.plot([H_thetas], [H_rhos], color="yellow", marker='o')
    sp.title.set_text("Hough rho and theta Space")
def graph_rectangles_and_circles(img, rectangles, circles, color=(0, 0, 255)):
    for box in rectangles:
        img = cv2.drawContours(img, box, 0, color , 6)
    for r,x,y in circles:
        img = cv2.circle(img, (y, x), r, color, 6)
    return img
def graph_output(edge_image, img_lines, img_w_rect_circles, hough_space, peaks):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15,15))
    sp1 = fig.add_subplot(2, 2, 1)
    sp1.imshow(edge_image, cmap='gray')
    sp1.title.set_text("Edge Image from Line Hough Transform")

    sp2 = fig.add_subplot(2, 2, 2)
    plot_hough_space(sp2, hough_space, peaks)

    sp3 = fig.add_subplot(2, 2, 3)
    sp3.title.set_text("Detected Lines")
    sp3.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))

    sp4 = fig.add_subplot(2, 2, 4)
    sp4.title.set_text("Detected Rectangles and Circles")
    sp4.imshow(cv2.cvtColor(img_w_rect_circles, cv2.COLOR_BGR2RGB))
    plt.show()

def countMoney(circles, areas):
    amount = 0
    for r,x, y in circles:
        if r <= 54:
            amount+=0.5
        else:
            amount+=1
    for _, height in areas:
        if height > 606:
            amount+=20
        else:
            amount+=1 
    return amount   

