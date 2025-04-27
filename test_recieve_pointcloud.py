import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model.pointnet import pointnet_model
from utils.dataloader import get_datasets

# Paths
CHECKPOINT_PATH = "checkpoints/best_model.h5"
HDF5_FILE_PATH = "robotic_shapes_preprocessed.h5"

# Hyperparameters
BATCH_SIZE = 16
NUM_CLASSES = 3
TARGET_CATEGORIES = [0, 1, 2]  # Change this to the desired categories

# Camera parameters
image_width = 200
image_height = 100
fov = 120  # Field of view in degrees
aspect = image_width / image_height
near_plane = 0.1
far_plane = 10.0
camera_position = [0.7, 0, 1.0]
target_position = [0.7, 0.2, 0.1]  # Looking at the cube
up_vector = [0, 0, 1]

# Connect to physics server
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and load the environment
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load a cube
cube_id = p.loadURDF("cube_small.urdf", [0.7, 0.3, 0.1], globalScaling=2)
p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

# Compute view and projection matrices
view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

# Simulate to let objects settle
for _ in range(240):
    p.stepSimulation()

# Get camera image (RGB + Depth)
_, _, rgb_image, depth_image, _ = p.getCameraImage(
    width=image_width,
    height=image_height,
    viewMatrix=view_matrix,
    projectionMatrix=projection_matrix
)

# Convert depth image to 3D point cloud
def depth_to_point_cloud(depth_image, projection_matrix, view_matrix, image_width, image_height):
    # Intrinsic camera parameters (focal lengths and center)
    fx = image_width / (2 * np.tan(np.radians(fov) / 2))
    fy = fx
    cx = image_width / 2
    cy = image_height / 2

    # Create an empty list for storing 3D points
    points = []

    # For each pixel, convert depth to 3D coordinates
    for i in range(image_height):
        for j in range(image_width):
            # Depth value at (i, j)
            z_cam = depth_image[i, j] * (far_plane - near_plane) + near_plane
            x_cam = (j - cx) * z_cam / fx
            y_cam = (i - cy) * z_cam / fy

            # Construct point in camera space
            point_cam = np.array([x_cam, y_cam, z_cam, 1])

            # Transform to world coordinates using the inverse of the view matrix
            point_world = np.linalg.inv(np.array(view_matrix).reshape(4, 4)) @ point_cam
            points.append(point_world[:3])

    return np.array(points)

# Downsample the point cloud to 1024 points
def downsample_point_cloud(point_cloud, num_points=1024):
    """
    Downsample the point cloud to a fixed number of points (e.g., 1024).
    """
    if point_cloud.shape[0] > num_points:
        # Randomly sample num_points points from the original point cloud
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        downsampled_point_cloud = point_cloud[indices]
    else:
        # If the point cloud has fewer points than num_points, pad with zeros
        padding = np.zeros((num_points - point_cloud.shape[0], 3))
        downsampled_point_cloud = np.vstack([point_cloud, padding])
    
    return downsampled_point_cloud

# Load the PointNet model
def load_model():
    model = pointnet_model(num_classes=NUM_CLASSES)
    model.load_weights(CHECKPOINT_PATH)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Visualize the point cloud in 3D and add prediction as text
def visualize_point_cloud_with_prediction(points, predicted_class, title="Point Cloud"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    # Add a descriptive label for the predicted class
    class_labels = {
        0: "Cube",
        1: "sphere",  # Example label for class 1
        2: "triangle"  # Example label for class 2
    }
    class_label = class_labels.get(predicted_class, f"Class {predicted_class}")
    
    ax.set_title(f"{title} - Predicted Class: {class_label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Add the predicted class as text on the plot
    ax.text2D(0.05, 0.95, f"Predicted Class: {class_label}", transform=ax.transAxes, fontsize=12, verticalalignment='top')

    plt.show()

# Test the model with point cloud input
def test_model_with_point_cloud(model, point_cloud):
    # Downsample the point cloud to match the model's expected input shape
    downsampled_point_cloud = downsample_point_cloud(point_cloud, num_points=1024)

    # Add batch dimension and make prediction
    predictions = model.predict(np.expand_dims(downsampled_point_cloud, axis=0))  # Add batch dimension
    predicted_class = np.argmax(predictions, axis=1)

    return predicted_class[0]


if __name__ == "__main__":
    # Get point cloud from depth image
    point_cloud = depth_to_point_cloud(depth_image, projection_matrix, view_matrix, image_width, image_height)

    # Load the model
    print("Rebuilding the model architecture...")
    model = load_model()

    # Get the predicted class
    predicted_class = test_model_with_point_cloud(model, point_cloud)

    # Visualize the point cloud with the predicted class
    visualize_point_cloud_with_prediction(point_cloud, predicted_class, title="Captured Point Cloud")

    # Keep the simulation running
    while True:
        p.stepSimulation()

    p.disconnect()
