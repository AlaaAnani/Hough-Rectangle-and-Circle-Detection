# Hough-Rectangle-and-Circle-Detection-from-Scratch
This is for a Computer Vision course assignment at AUC. This Python project purely coded using Numpy takes a money image, determines the rectangles and circles in it and also counts the money [Calibrated on the provided test set only].

# Hough Line Transform
The way lines are detected is through the following steps:

1. Apply medianBlur filter (7, 7)

2. Apply GaussianBlur filter (7, 7)

3. Apply bilateralFilter 
(The previous steps are to get rid of the noise before doing edge detection)

4. Run Canny edge detector

5. Create the hough space of <img src="https://render.githubusercontent.com/render/math?math=\rho"> and <img src="https://render.githubusercontent.com/render/math?math=\theta"> with increments of <img src="https://render.githubusercontent.com/render/math?math=\Delta \rho"> and <img src="https://render.githubusercontent.com/render/math?math=\Delta \theta"> speicfied from the arguements `num_rhos` and `num_thetas`. 

6. Retreive (x, y) values of the edges in the image resulting from Canny.

7. Calculate <img src="https://render.githubusercontent.com/render/math?math=\rho"> values by multiplying the matrices <img src="https://render.githubusercontent.com/render/math?math=(x, y) * (cos(\theta), sin(\theta))">

8. Build a histogram of the accumulator space and filter the lines based on the `threshold` specified.

9. Filter similar lines (Lines whose difference in <img src="https://render.githubusercontent.com/render/math?math=\rho"> and <img src="https://render.githubusercontent.com/render/math?math=\theta$">
# Rectangles Detection from Lines Resulting from Hough Transform

1. Find all pairs of parallel lines. A pair of lines <img src="https://render.githubusercontent.com/render/math?math=H_1(\rho_1, \theta_2)"> and <img src="https://render.githubusercontent.com/render/math?math=H_2(\rho_2, \theta_2)"> are parallel if <img src="https://render.githubusercontent.com/render/math?math=abs(\theta_1 - \theta_2) < \epsilon_{\theta}">

2. After finding all pairs of <img src="https://render.githubusercontent.com/render/math?math=H_i"> and <img src="https://render.githubusercontent.com/render/math?math=H_j"> statisfying the parallel condition, this generates what is called extended peaks <img src="https://render.githubusercontent.com/render/math?math=P(\alpha, \zeta)"> where:

<img src="https://render.githubusercontent.com/render/math?math=\alpha = (\theta_i + theta_j)/2">

<img src="https://render.githubusercontent.com/render/math?math=\zeta = (\rho_i - rho_j)/2">

3. By looping over all <img src="https://render.githubusercontent.com/render/math?math=P_i">s found, we need to find peaks that are perpedindecular to each other. <img src="https://render.githubusercontent.com/render/math?math=P_k"> and <img src="https://render.githubusercontent.com/render/math?math=P_l"> are perpedindecular iff:

<img src="https://render.githubusercontent.com/render/math?math=\Delta \alpha = ||\alpha_k - \alpha_l| - 90| < \epsilon_{\alpha}"> where <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{\alpha}"> is the angular threshold that determines whether two <img src="https://render.githubusercontent.com/render/math?math=P_i">s are perspendicular or not. 

In the code, <img src="https://render.githubusercontent.com/render/math?math=\epsilon_{\theta}=\epsilon_{\alpha}=\ang{3}">

4. The found 4 lines forming every rectangle are then passed to `get_lines_intersection` function which returns a point of intersection given two lines defined by 2 points each. The relation found here: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line is used to return the intersection point.

5. After getting the 4 bounding points for the rectangle, the function `minAreaReact` is used to return the correct format for a rectangle bounded by those points, which is then passed to `drawContours` to draw the rectangle on the image.

6. Only rectangles with `area` < `area_threshold` are considered valid, given that the `area_threshold` is tuned for the usecase of paper money in this assignment. 

# Circle Hough Transform

1. I build a 3D space of <img src="https://render.githubusercontent.com/render/math?math=(a, b, \theta)">, where a circle is defined as the following:

<img src="https://render.githubusercontent.com/render/math?math=x = a + r*cos(\theta)">

<img src="https://render.githubusercontent.com/render/math?math=y = b + r*sin(\theta)">

The flow is similar to the line hough transform except is scaled to searching a third dimension specified by the `r_range` for the radius.

# BONUS: Money Count

I count the money purely based on the area of the paper or the radius of the coin. 

The coins I experimented with are 0.5 and 1 pound (Which are very close in radii, so the threshold isn't always accurate)

The papers I count are: 
1. 1-pound paper

2. 20-pound (based on its height, which is way more than the 1-pound paper)

The program is limited to those since this is what my test set is composed of, but it can definitely be scaled to all egyptian money after experimting with them and knowing their width and height.
