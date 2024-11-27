import numpy as np
from typing import List, Optional, Tuple
import itertools
from sklearn.decomposition import PCA

"""
Based on Iterative Hough Transform for Line Detection in 3D Point Clouds: https://doi.org/10.5201/ipol.2017.208
"""

def discretize_sphere(
    tesselation_steps: int
) -> np.ndarray:
    
    """
    Discretizes latent space for values of b, 
    where b is the b parameter in a + tb line representation for 3d
    by tesselating an icosahedron

    Args:
        tesselation_steps: number of iterations 

    Returns:
        Numpy array of b values
    """

    tau = (1 + np.sqrt(5)) / 2
    norm = np.sqrt(1 + tau * tau)
    v = 1 / norm
    tau = tau / norm

    vertices = [
        np.array([-v,tau,0]),
        np.array([v,tau,0]),
        np.array([0,v,-tau]),
        np.array([0,v,tau]),
        np.array([-tau,0,-v]),
        np.array([tau,0,-v]),
        np.array([-tau,0,v]),
        np.array([tau,0,v]),
        np.array([0,-v,-tau]),
        np.array([0,-v,tau]),
        np.array([-v,-tau,0]),
        np.array([v,-tau,0])
    ]

    triangles = [
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[4]],
        [vertices[0], vertices[4], vertices[6]],
        [vertices[0], vertices[3], vertices[6]],
        [vertices[1], vertices[2], vertices[5]],
        [vertices[1], vertices[3], vertices[7]],
        [vertices[1], vertices[5], vertices[7]],
        [vertices[2], vertices[4], vertices[8]],
        [vertices[2], vertices[5], vertices[8]],
        [vertices[3], vertices[6], vertices[9]],
        [vertices[3], vertices[7], vertices[9]],
        [vertices[4], vertices[8], vertices[10]],
        [vertices[8], vertices[10], vertices[11]],
        [vertices[5], vertices[8], vertices[11]],
        [vertices[5], vertices[7], vertices[11]],
        [vertices[7], vertices[9], vertices[11]],
        [vertices[9], vertices[10], vertices[11]],
        [vertices[6], vertices[9], vertices[10]],
        [vertices[4], vertices[6], vertices[10]],
    ]

    def middle_point(first_vertex: np.ndarray, second_vertex: np.ndarray) -> np.ndarray:
        middle_point = (first_vertex + second_vertex) / np.linalg.norm(first_vertex + second_vertex)
        return middle_point

    for _ in range(tesselation_steps):
        new_triangles = []

        for triangle in triangles:
            
            """
                0
               / \
              1 - 2

               into

                0
               / \
              3 - 4
             / \ / \
            1 - 5 - 2
            """

            v_0, v_1, v_2 = triangle
            v_3 = middle_point(v_0, v_1)
            v_4 = middle_point(v_0, v_2)
            v_5 = middle_point(v_1, v_2)

            vertices.append(v_3)
            vertices.append(v_4)
            vertices.append(v_5)

            new_triangles.append([v_0, v_3, v_4])
            new_triangles.append([v_3, v_1, v_5])
            new_triangles.append([v_5, v_4, v_3])
            new_triangles.append([v_4, v_5, v_2])
        
        # append all new points
        triangles = new_triangles
    
    unique_vertices = np.unique(np.array(vertices), axis=0)
    upper_hemisphere = unique_vertices[:, 2] > 0
    equator = unique_vertices[:, 2] == 0
    left_hemisphere = unique_vertices[:, 1] >= 0
    without_left_hemisphere_duplicate = unique_vertices[:, 0] != -1

    return unique_vertices[upper_hemisphere | (equator & left_hemisphere & without_left_hemisphere_duplicate)]

def plane_to_point(x_p: float, y_p: float, b: np.ndarray) -> np.ndarray:
    """
    Restores the anchor point a for line in form of a + tb given x', y' and b
    """
    
    b_x, b_y, b_z = b
    beta = 1 / (1 + b_z)

    x = x_p * (1 - ((b_x * b_x) * beta)) - y_p * ((b_x * b_y) * beta)
    y = x_p * (-((b_x * b_y) * beta)) + y_p * (1 - ((b_y * b_y) * beta))
    z = (-x_p) * b_x - y_p * b_y

    return np.array([x, y, z])

def point_to_plane(b, p):
    """
    Finds the point of intersection (x', y') of the line p + tb with the plane
    given point and vector b
    """

    b_x, b_y, b_z = b
    p_x, p_y, p_z = p

    beta = 1 / (1 + b_z)

    x = ((1 - (beta * (b_x * b_x))) * p_x) - ((beta * (b_x * b_y)) * p_y) - (b_x * p_z)
    y = ((-beta * (b_x * b_y)) * p_x) + ((1 - (beta * (b_y * b_y))) * p_y) - (b_y * p_z)

    return x, y

def point_to_plane_batched(b, p):
    """
    Finds the points of intersection (x', y') of the lines p + tb with the plane
    given point and vectors b
    """

    b_x, b_y, b_z = b[:, 0], b[:, 1], b[:, 2]
    p_x, p_y, p_z = p

    beta = 1 / (1 + b_z)

    xs = ((1 - (beta * (b_x * b_x))) * p_x) - ((beta * (b_x * b_y)) * p_y) - (b_x * p_z)
    ys = ((-beta * (b_x * b_y)) * p_x) + ((1 - (beta * (b_y * b_y))) * p_y) - (b_y * p_z)

    return xs, ys

def calculate_optimal_delta(points: np.ndarray) -> np.ndarray:
    """
    Calculates optimal discretization given the points
    """

    return np.linalg.norm(np.max(points, axis=0) - np.min(points, axis=0)) / 64

def partition_points(
    points: np.ndarray,
    line: Tuple[np.ndarray, np.ndarray], 
    delta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partitions point cloud based on the distance from a line
    """

    a, b = line
    
    t = np.dot((points - a), b)
    projections = a + np.outer(t, b)
    distances = np.linalg.norm(points - projections, axis=1)

    close_points = points[distances <= delta]
    far_points = points[distances > delta]
    
    return close_points, far_points

def spatial_hough(
    points: np.ndarray,
    tesselation_steps: int = 4,
    delta: Optional[float] = None,
    threshold: int = 50,
    adjust_lines: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Performs hough transform on 3d data

    Args:
        points: numpy array of point data
        tesselation_steps: discretization steps for the latent space of parameter b
        delta: discretization step for the plane that is used to restore parameter a
        threshold: number of points on the line to consider it candidate
        adjust_lines: whether to apply least squares fit to avoid discretization inaccuracies

    Returns:
        List of lines in a form (a, b)
    """
    
    directions = discretize_sphere(tesselation_steps)

    center = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - center

    max_shifted = np.linalg.norm(np.max(points, axis=0))
    min_shifted = np.linalg.norm(np.min(points, axis=0))
    half_norm = np.max([max_shifted, min_shifted])

    if delta is None:
        delta = calculate_optimal_delta(points)

    xs = np.arange(-half_norm, half_norm, delta)
    ys = np.arange(-half_norm, half_norm, delta)
    array = np.zeros([xs.shape[0], ys.shape[0], directions.shape[0]])

    directions_indices = np.arange(0, len(directions), 1, dtype=int)

    def hough_aux(points, inverse = False):        
        for p in points:
            xs, ys = point_to_plane_batched(directions, p)
                
            xs = ((xs + half_norm) // delta).astype(int)
            ys = ((ys + half_norm) // delta).astype(int)
            indices = np.stack((xs, ys, directions_indices))

            vote = -1 if inverse else 1
            array[indices[0], indices[1], indices[2]] += vote        
                

    hough_aux(points)
    lines = []

    while True:
        best_line_index = np.unravel_index(np.argmax(array), array.shape)

        x = xs[best_line_index[0]]
        y = ys[best_line_index[1]]
        b = directions[best_line_index[2]]
        a = plane_to_point(x, y, b)

        best_line = (a, b)
        close_points, far_points = partition_points(points, best_line, delta)

        if len(close_points) < threshold:
            return lines

        if adjust_lines:
            a_opt = np.mean(close_points, axis=0)
            b_opt = PCA(n_components=1).fit(close_points).components_[0]

            optimal_line = (a_opt, b_opt)
            best_line = optimal_line

        close_points, far_points = partition_points(points, best_line, delta)
        points = far_points

        hough_aux(close_points, inverse=True)

        best_line = (best_line[0] + center, best_line[1])
        lines.append(best_line)

        if len(close_points) < threshold or len(far_points) < threshold:
            return lines