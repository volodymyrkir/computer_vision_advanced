import plotly.graph_objects as go

import numpy as np
import pandas as pd

MIN_OBJECTS, MAX_OBJECTS = 17, 20
MIN_LINE_LENGTH, MAX_LINE_LENGTH = 5, 25
NUM_POINTS_PER_OBJECT = 50
SPACE_BOUNDARY = 100
TIME_LIMIT = 10
LINE_NOISE_INTENSITY = 0.2
FAKE_DETECTIONS = 1300


def generate_moving_objects(
    min_objects: int = MIN_OBJECTS,
    max_objects: int = MAX_OBJECTS,
    min_line_length: int = MIN_LINE_LENGTH,
    max_line_length: int = MAX_LINE_LENGTH,
    num_points_per_object: int = NUM_POINTS_PER_OBJECT,
    space_boundary: int = SPACE_BOUNDARY,
    time_limit: int = TIME_LIMIT,
    line_noise_intensity: float = LINE_NOISE_INTENSITY,
    fake_detections: int = FAKE_DETECTIONS,
) -> pd.DataFrame:
    np.random.seed(42)
    dataset = []
    num_objects = np.random.randint(min_objects, max_objects + 1)
    for obj_id in range(1, num_objects + 1):
        line_length = np.random.uniform(min_line_length, max_line_length)

        start_point = np.random.uniform(0, space_boundary, size=3)
        direction = np.random.uniform(-1, 1, size=3)
        direction /= np.linalg.norm(direction)  # Normalize direction vector

        for t in np.linspace(0, time_limit, num_points_per_object):
            displacement = direction * t * (line_length / time_limit)
            point = start_point + displacement
            noisy_point = point + np.random.normal(0, line_noise_intensity, size=3)
            dataset.append([t, *noisy_point, obj_id])

    # Generate fake detections
    for _ in range(fake_detections):
        t_fake = np.random.uniform(0, time_limit)
        x_fake = np.random.uniform(0, space_boundary)
        y_fake = np.random.uniform(0, space_boundary)
        z_fake = np.random.uniform(0, space_boundary)
        dataset.append([t_fake, x_fake, y_fake, z_fake, -1])  # -1 indicates a fake object

    columns = ["time", "x", "y", "z", "object_id"]
    df = pd.DataFrame(dataset, columns=columns)
    return df


def visualize_3d_interactive(df: pd.DataFrame) -> None:
    fig = go.Figure()
    for obj_id in df["object_id"].unique():
        subset = df[df["object_id"] == obj_id]
        if obj_id == -1:
            fig.add_trace(go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode='markers',
                marker=dict(size=5, color='red', opacity=0.7),
                name="Fake Detections"
            ))
        else:
            fig.add_trace(go.Scatter3d(
                x=subset["x"],
                y=subset["y"],
                z=subset["z"],
                mode='lines+markers',
                marker=dict(size=4),
                name=f"Object {int(obj_id)}"
            ))

    fig.update_layout(
        title="3D Visualization of Moving Objects and Fake Detections",
        scene=dict(
            xaxis_title="X Coordinate",
            yaxis_title="Y Coordinate",
            zaxis_title="Z Coordinate"
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.1, y=0.9)
    )

    fig.show()


if __name__ == '__main__':
    data = generate_moving_objects()
    print(data.head())
    visualize_3d_interactive(data)
