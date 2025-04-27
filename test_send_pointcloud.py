import pybullet as p
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

# Connect to physics server
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity and load the environment
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# Load a cube
cube_id = p.loadURDF("cube_small.urdf", [0.7, 0.3, 0.1], globalScaling=2)
p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

# Camera parameters
image_width = 200
image_height = 100
fov = 120  # Field of view in degrees
aspect = image_width / image_height
near_plane = 0.1
far_plane = 10.0

# Camera position and orientation
camera_position = [0.6, 0, 1.0]
target_position = [0.7, 0.2, 0.1]  # Looking at the cube
up_vector = [0, 0, 1]

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

# Get point cloud from depth image
point_cloud = depth_to_point_cloud(depth_image, projection_matrix, view_matrix, image_width, image_height)

# Save point cloud to a file (optional)
np.save("point_cloud.npy", point_cloud)

# Plot the point cloud in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
if len(point_cloud) > 0:
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='r', marker='o', s=1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Point Cloud from Depth Image')
plt.show()

# Send the point cloud data to the test script
def evaluate_point_cloud(point_cloud):
    # Load your trained model
    model = tf.keras.models.load_model('checkpoints/best_model.h5')

    # Preprocess the point cloud for classification (this depends on your model)
    point_cloud = point_cloud.astype(np.float32)  # Ensure data type is correct
    point_cloud = np.expand_dims(point_cloud, axis=0)  # Add batch dimension if needed

    # Make predictions
    predictions = model.predict(point_cloud)
    predicted_class = np.argmax(predictions, axis=1)
    print(f"Predicted Class: {predicted_class[0]}")

# Call the evaluation function
evaluate_point_cloud(point_cloud)

# Keep the simulation running
while True:
    p.stepSimulation()

p.disconnect()
