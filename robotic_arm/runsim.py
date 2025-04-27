import pybullet as p
import pybullet_data
import time

# Connect to the GUI
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Setup world
p.loadURDF("plane.urdf")
p.setGravity(0, 0, -9.81)

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

# Configuration variables
gripper_link_index = 8  # Ensure this is correct!
attached = False
constraint_id = None
move_stage = 0

# Main simulation loop
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
                maxVelocity=0.5
            )

    # Attach the cube after reaching the first target position
    if move_stage == 0:
        current_pos = p.getLinkState(robot_id, gripper_link_index)[0]
        distance_to_target = sum((current_pos[i] - target_pos[i]) ** 2 for i in range(3))
        if distance_to_target < 0.01:  # Check if the robot is close enough to the cube
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
                parentFramePosition=[0, 0, 0],  # Relative position of the cube w.r.t. the gripper
                childFramePosition=[0, 0, 0],   # Cube's local frame
                parentFrameOrientation=[0, 0, 0, 1],  # Align orientations
                childFrameOrientation=[0, 0, 0, 1]
            )

            attached = True
            move_stage = 1  # Move to the next stage (lifting the cube)

    # Detachment logic
    if attached and move_stage == 1:
        current_pos = p.getLinkState(robot_id, gripper_link_index)[0]
        target = [0.2, 0.2, 0.5]
        distance_to_target = sum((current_pos[i] - target[i]) ** 2 for i in range(3))
        if distance_to_target < 0.01:  # Check if the robot has reached the second target position
            p.removeConstraint(constraint_id)  # Remove the fixed constraint to detach the cube
            p.resetBaseVelocity(cube_id, [0, 0, 0], [0, 0, 0])  # Stop the cube from moving
            attached = False
            move_stage = 2  # End the sequence

    p.stepSimulation()
    time.sleep(1.0 / 240.0)

p.disconnect()