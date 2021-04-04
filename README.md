# Hough-Rectangle-and-Circle-Detection-from-Scratch
This is for a Computer Vision course assignment at AUC. This Python project purely coded using Numpy takes a money image, determines the rectangles and circles in it and also counts the money [Calibrated on the provided test set only].

# Hough Line Transform
The way lines are detected is through the following steps:

1. Apply medianBlur filter (7, 7)

2. Apply GaussianBlur filter (7, 7)

3. Apply bilateralFilter 
(The previous steps are to get rid of the noise before doing edge detection)

4. Run Canny edge detector

5. Create the hough space of $\rho$ and $\theta$ with increments of $\Delta \rho$ and $\delta \theta$ speicfied from the arguements `num_rhos` and `num_thetas`. 

6. Retreive (x, y) values of the edges in the image resulting from Canny.

7. Calculate $\rho$ values by multiplying the matrices $(x, y) * (cos(\theta), sin(\theta))$

8. Build a histogram of the accumulator space and filter the lines based on the `threshold` specified.

9. Filter similar lines (Lines whose difference in $\rho$ and $\theta$ is less than the `lines_similarity_threshold` ), which I tuned by experimenting.

# Rectangles Detection from Lines Resulting from Hough Transform

1. Find all pairs of parallel lines. A pair of lines $H_1(\rho_1, \theta_2)$ and $H_2(\rho_2, \theta_2)$ are parallel if $abs(\theta_1 - \theta_2) < \epsilon_{\theta}$

2. After finding all pairs of $H_i$ and $H_j$ statisfying the parallel condition, this generates what is called extended peaks $P(\alpha, \zeta)$ where:

$\alpha = (\theta_i + theta_j)/2$

$\zeta = (\rho_i - rho_j)/2$

3. By looping over all $P_i$s found, we need to find peaks that are perpedindecular to each other. $P_k$ and $P_l$ are perpedindecular iff:

$\Delta \alpha = ||\alpha_k - \alpha_l| - 90| < \epsilon_{\alpha}$ where $\epsilon_{\alpha}$ is the angular threshold that determines whether two $P_i$s are perspendicular or not. 

In the code, $\epsilon_{\theta}=\epsilon_{\alpha}=3 degrees$

4. The found 4 lines forming every rectangle are then passed to `get_lines_intersection` function which returns a point of intersection given two lines defined by 2 points each. The relation found here: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line is used to return the intersection point.

5. After getting the 4 bounding points for the rectangle, the function `minAreaReact` is used to return the correct format for a rectangle bounded by those points, which is then passed to `drawContours` to draw the rectangle on the image.

6. Only rectangles with `area` < `area_threshold` are considered valid, given that the `area_threshold` is tuned for the usecase of paper money in this assignment. 

# Circle Hough Transform

1. I build a 3D space of $(a, b, \theta)$, where a circle is defined as the following:

$x = a + r*cos(\theta)$

$y = b + r*sin(\theta)$

The flow is similar to the line hough transform except is scaled to searching a third dimension specified by the `r_range` for the radius.

# BONUS: Money Count

I count the money purely based on the area of the paper or the radius of the coin. 

The coins I experimented with are 0.5 and 1 pound (Which are very close in radii, so the threshold isn't always accurate)

The papers I count are: 
1. 1-pound paper

2. 20-pound (based on its height, which is way more than the 1-pound paper)

The program is limited to those since this is what my test set is composed of, but it can definitely be scaled to all egyptian money after experimting with them and knowing their width and height.
