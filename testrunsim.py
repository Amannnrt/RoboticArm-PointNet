import pybullet as p
import pybullet_data
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from model.pointnet import pointnet_model
import time

# Paths
CHECKPOINT_PATH = "checkpoints/best_model.h5"

# Hyperparameters
NUM_CLASSES = 3

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
p.setGravity(0, 0, -11)
p.loadURDF("plane.urdf")

# Load robot
robot_id = p.loadURDF(
    "C://Users//iammd//Desktop//pointnet-robotic_arm//robotic_arm//urdf//lbr_iiwa_14_r820.urdf",
    [0, 0, 0],
    useFixedBase=True,
)

# Load cube with increased size
cube_start_pos = [0.7, 0.3, 0.1]
cube_id = p.loadURDF("cube_small.urdf", cube_start_pos, globalScaling=1.5)
p.changeVisualShape(cube_id, -1, rgbaColor=[1, 0, 0, 1])

# Compute view and projection matrices
view_matrix = p.computeViewMatrix(camera_position, target_position, up_vector)
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane, far_plane)

# Configuration variables for the robot
gripper_link_index = 8
attached = False
constraint_id = None
move_stage = 0

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

# Convert depth image to 3D point cloud
def depth_to_point_cloud(depth_image, projection_matrix, view_matrix, image_width, image_height):
    fx = image_width / (2 * np.tan(np.radians(fov) / 2))
    fy = fx
    cx = image_width / 2
    cy = image_height / 2

    points = []
    for i in range(image_height):
        for j in range(image_width):
            z_cam = depth_image[i, j] * (far_plane - near_plane) + near_plane
            x_cam = (j - cx) * z_cam / fx
            y_cam = (i - cy) * z_cam / fy
            point_cam = np.array([x_cam, y_cam, z_cam, 1])
            point_world = np.linalg.inv(np.array(view_matrix).reshape(4, 4)) @ point_cam
            points.append(point_world[:3])

    return np.array(points)

# Downsample the point cloud to 1024 points
def downsample_point_cloud(point_cloud, num_points=1024):
    if point_cloud.shape[0] > num_points:
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        downsampled_point_cloud = point_cloud[indices]
    else:
        padding = np.zeros((num_points - point_cloud.shape[0], 3))
        downsampled_point_cloud = np.vstack([point_cloud, padding])
    return downsampled_point_cloud

# Test the model with point cloud input
def test_model_with_point_cloud(model, point_cloud):
    downsampled_point_cloud = downsample_point_cloud(point_cloud, num_points=1024)
    predictions = model.predict(np.expand_dims(downsampled_point_cloud, axis=0))
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Visualize the point cloud in 3D and add prediction as text
def visualize_point_cloud_with_prediction(points, predicted_class, title="Point Cloud"):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    
    class_labels = {
        0: "Cube",
        1: "Sphere",
        2: "Triangle"
    }
    class_label = class_labels.get(predicted_class, f"Class {predicted_class}")
    
    ax.set_title(f"{title} - Predicted Class: {class_label}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    # ax.text2D(0.05, 0.95, f"Predicted Class: {class_label}", transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.show()

# Main simulation loop
if __name__ == "__main__":
    # Load the PointNet model
    print("Rebuilding the model architecture...")
    model = load_model()

    while True:
        # Determine target based on stage
        target_pos = None
        if move_stage == 0:
            target_pos = [0.7, 0.3, 0.1]  # Position near the cube
        elif move_stage == 1:
            target_pos = [0.2, 0.2, 0.5]  # Lift and move the cube

        # Continuous IK updates for smooth movement
        if target_pos:
            joint_positions = p.calculateInverseKinematics(
                robot_id, gripper_link_index, target_pos,
                maxNumIterations=100,
                residualThreshold=0.001
            )
            for i in range(7):
                p.setJointMotorControl2(
                    robot_id, i, p.POSITION_CONTROL,
                    joint_positions[i],
                    force=500,
                    maxVelocity=0.7
                )

        # Attach the cube after reaching the first target position
        if move_stage == 0:
            current_pos = p.getLinkState(robot_id, gripper_link_index)[0]
            distance_to_target = sum((current_pos[i] - target_pos[i]) ** 2 for i in range(3))
            if distance_to_target < 0.01:
                gripper_pos, gripper_orn = p.getLinkState(robot_id, gripper_link_index)[0:2]
                cube_pos, cube_orn = p.getBasePositionAndOrientation(cube_id)

                # Create a fixed constraint to attach the cube to the gripper
                constraint_id = p.createConstraint(
                    parentBodyUniqueId=robot_id,
                    parentLinkIndex=gripper_link_index,
                    childBodyUniqueId=cube_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=[0, 0, 0, 1]
                )

                attached = True
                move_stage = 1  

                
                _, _, rgb_image, depth_image, _ = p.getCameraImage(
                    width=image_width,
                    height=image_height,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix
                )
                point_cloud = depth_to_point_cloud(depth_image, projection_matrix, view_matrix, image_width, image_height)
                predicted_class = test_model_with_point_cloud(model, point_cloud)
                visualize_point_cloud_with_prediction(point_cloud, predicted_class, title="Captured Point Cloud")

       
        if attached and move_stage == 1:
            current_pos = p.getLinkState(robot_id, gripper_link_index)[0]
            target = [0.2, 0.3, 0.5]
            distance_to_target = sum((current_pos[i] - target[i]) ** 2 for i in range(3))
            if distance_to_target < 0.06:
                p.removeConstraint(constraint_id)
                p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0])
                attached = False
                move_stage = 2  

        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()

# //////////////////////////////////////////////////////////////////////////////////////////


