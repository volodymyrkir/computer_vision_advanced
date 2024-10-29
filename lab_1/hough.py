import numpy as np
import cv2
import typing as tp
from skimage.filters import gaussian

def hough_coordinates(
    image: np.ndarray,
    rho: int | float = 1,
    theta: int | float = np.pi / 180,
    threshold: int | float = 60.0,
) -> np.ndarray:
    """
    Performs standard hough transform: 
        Creates grid in hough (straight line) parameter space
        Accumulates scores for each line candidate by counting points that lay on corresponding lines

    Args:
        image: numpy array of pixel intensities
        rho: size of accumulated array cell on rho axis
        theta: size of accumulated array cell on theta axis 
        threshold: threshold to consider the activated cell a candidate

    Returns:
        Numpy array of accumulated values
    """

    diagonal = np.linalg.norm(np.array(image.shape))
    angles = np.arange(-np.pi/2, np.pi/2, theta)
    rho_values = np.arange(-diagonal, diagonal, rho)

    width, height = 2 * diagonal, np.pi
    rho_steps = int(width / rho)
    theta_steps = int(height / theta)

    accumulator = np.zeros([rho_steps, theta_steps])    

    sin_values = np.sin(angles)
    cos_values = np.cos(angles)

    indices = np.indices(image.shape)
    rows, cols = indices

    rho_indices = np.outer(cols, cos_values) + np.outer(rows, sin_values)
    rho_indices = ((rho_indices + diagonal) / rho).astype(int)

    theta_indices = np.repeat(np.arange(theta_steps), rho_indices.shape[0])
    theta_indices = np.reshape(theta_indices, rho_indices.shape, order='F')
    
    indices_mapping = np.repeat(np.expand_dims(indices, -1), theta_steps, -1)
    values = image[indices_mapping[0], indices_mapping[1]]
    values = np.reshape(values, rho_indices.shape)

    np.add.at(accumulator, (rho_indices, theta_indices), values)
        
    detected_rho_index, detected_theta_index = np.where(accumulator > threshold)
    detected_rho = rho_values[detected_rho_index]    
    detected_theta = angles[detected_theta_index]
    
    polar_coordinates = np.vstack([detected_rho, detected_theta]).T
    return polar_coordinates

def polar2cartesian(radius: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Converts candidate points to line parameters in cartesian space

    Args: 
        radius: rho values, distance from the start of coordinates 
        angle: theta values

    Returns:
        points of intersection between encoded line and normal from the center of coordinates
    """
    return radius * np.array([np.cos(angle), np.sin(angle)])

def clusterize_coordinates(
    coordinates: np.ndarray,
    rho: int | float,
    theta: int | float,
    overlap_threshold: int | float = 5
) -> np.ndarray:
    """
    Receives raw "list" of detected lines and clusterizes lines that are too similar 

    Args:
        coordinates: candidate cells coordinates after accumulation
        rho: value of rho parameter used by transform
        theta: value of theta parameter used by transform
        overlap_threshold: delta (in accumulator cell sides) that is considered an overlap

    Returns:
        Averaged lines per each found cluster 
    """
    
    # Lines are clustered naively - DBSCAN may do better

    rho_threshold = overlap_threshold * rho
    theta_threshold = overlap_threshold * theta
    groups = []

    def close(first_line: np.ndarray, second_line: np.ndarray) -> bool:
        return np.all(np.abs(first_line - second_line) < np.array([rho_threshold, theta_threshold]))

    for line in coordinates:
        found_cluster = False
        for group in groups:
            if any(close(l, line) for l in group):
                group.append(line)
                found_cluster = True
                break
        
        if not found_cluster:
            new_group = [line]
            groups.append(new_group)
    
    return np.array(list(map(lambda g: np.average(g, axis=0), groups)))
    

def coordinates_to_lines(
    image: np.ndarray,
    coordinates: np.ndarray,
    rho: int | float,
    theta: int | float,
    overlap_threshold: int | float = 5,
    min_line_length: int | float = 50,
    max_line_gap: int | float = 5, 
) -> np.ndarray:
    """
    Converts candidate cells from accumulated array to corresponding lines
    Performs validation on resulting lines
    
    Args:
        image: input image as array of intensities
        coordinates: candidate cells coordinates after accumulation
        rho: value of rho parameter used by transform
        theta: value of theta parameter used by transform
        overlap_threshold: delta (in accumulator cell sides) that is considered an overlap
        min_line_length: minimal acceptable size of the line segment, used when segment bounds are calculated
        max_line_gap: maximal gap that can be considered an inconsistency rather than end of the segment
    
    Returns:
        Array of coordinates that describe ends of line segments found in the image
    """

    lines = []

    if len(coordinates.shape) == 1:
        coordinates = coordinates[None, ...]

    coordinates = clusterize_coordinates(coordinates, rho, theta, overlap_threshold)
    image_averaged = (gaussian(image * 255, 3) > 0.15).T

    for rho, theta in coordinates:
        pt0 = polar2cartesian(rho, theta)

        direction = np.array([-pt0[1], pt0[0]])
        direction = direction / np.linalg.norm(direction)

        xs = np.arange(image.shape[0])
        ys = -(np.cos(theta)/np.sin(theta)) * xs + rho / np.sin(theta)

        x1_candidates = np.argwhere((ys < image.shape[1]) & (ys >= 0))
        x1 = int(x1_candidates[0]) if x1_candidates.shape[0] > 0 else None

        if x1 is None:
            continue

        x2_candidates = np.argwhere((ys[x1:] > image.shape[1]) | (ys[x1:] < 0))
        x2 = (x1 + int(x2_candidates[0])) if x2_candidates.shape[0] > 0 else image.shape[0]

        x_bounds = np.array([x1-1, x2+1])

        current_segment = []
        current_gap = 0

        for x in np.arange(x_bounds[0], x_bounds[1], 1).astype(int):
            y = -(np.cos(theta)/np.sin(theta)) * x + rho / np.sin(theta)
            y = y.astype(int)

            if (x < 0 or x >= image.shape[0]) or (y < 0 or y >= image.shape[1]):
                continue

            contains_point = image_averaged[x, y]

            if contains_point:
                current_segment.append(np.array([x, y]))
                current_gap = 0
            else:
                current_gap += 1
            
            if current_gap > max_line_gap:
                if len(current_segment) >= min_line_length:
                    lines.append((current_segment[0][0], current_segment[0][1], current_segment[-1][0], current_segment[-1][1]))

                current_segment = []         

        if len(current_segment) >= min_line_length:
            lines.append((current_segment[0][0], current_segment[0][1], current_segment[-1][0], current_segment[-1][1]))

    return lines
    

def hough(
    image: np.ndarray,
    rho: int | float = 1,
    theta: int | float = np.pi/180,
    threshold: int | float = 60,
    overlap_threshold: int | float = 5,
    min_line_length: int | float = 50,
    max_line_gap: int | float = 5,
) -> np.ndarray:
    """
    Performs standard hough transform and returns detected line segments based on the results
    Utilizes implementation from this module

    Args:
        image: numpy array of pixel intensities
        rho: size of accumulated array cell on rho axis
        theta: size of accumulated array cell on theta axis 
        threshold: threshold to consider the activated cell a candidate
        overlap_threshold: delta (in accumulator cell sides) that is considered an overlap
        min_line_length: minimal acceptable size of the line segment, used when segment bounds are calculated
        max_line_gap: maximal gap that can be considered an inconsistency rather than end of the segment
    
    Returns:
        Array of coordinates that describe ends of line segments found in the image
    """

    coordinates = hough_coordinates(image, rho, theta, threshold)
    return coordinates_to_lines(image, coordinates, rho, theta, overlap_threshold, min_line_length, max_line_gap)

def hough_cv(
    image: np.ndarray,
    rho: int | float = 1,
    theta: int | float = np.pi/180,
    threshold: int | float = 60,
    overlap_threshold: int | float = 5,
    min_line_length: int | float = 50,
    max_line_gap: int | float = 5,
) -> np.ndarray:
    """
    Performs standard hough transform and returns detected line segments based on the results
    Utilizes implementation from cv2 module

    Args:
        image: numpy array of pixel intensities
        rho: size of accumulated array cell on rho axis
        theta: size of accumulated array cell on theta axis 
        threshold: threshold to consider the activated cell a candidate
        overlap_threshold: delta (in accumulator cell sides) that is considered an overlap
        min_line_length: minimal acceptable size of the line segment, used when segment bounds are calculated
        max_line_gap: maximal gap that can be considered an inconsistency rather than end of the segment
    
    Returns:
        Array of coordinates that describe ends of line segments found in the image
    """

    coordinates = cv2.HoughLines(image, rho=rho, theta=theta, threshold=threshold)    
    coordinates = np.squeeze(coordinates)
    return coordinates_to_lines(image, coordinates, rho, theta, overlap_threshold, min_line_length, max_line_gap)

