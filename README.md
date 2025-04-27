# Robotic Arm with PointCloud Classification and Manipulation

## Project Overview
This project integrates a robotic arm with a deep learning model (PointNet) for object classification and basic manipulation. The main goal was to implement a system where the robotic arm picks up and moves an object based on its classification.

### Key Steps:
- **PointCloud Classification**: The project uses PointNet to classify objects based on their 3D point cloud data. A depth camera is used to capture the point cloud, which is then fed into the PointNet model for classification.
  
- **Robotic Arm Manipulation**: After classifying an object, the robotic arm moves toward it, picks it up (with a simplified approach), and moves it to a predefined location.

## Limitations and Issues:
- **Gripper Issue**: The gripper's functionality is not fully resolved. Due to issues with proper attachment/detachment mechanics, I used a workaround by attaching the object directly upon contact.
  
- **Model Accuracy**: The PointNet model is inconsistent in its predictions. This is because I had to generate my own data for training, and the model was not trained well enough. With more data and proper training, the accuracy could improve. The model from this project achieves decent results but isn’t highly reliable.
  
- **Separate Classification Repository**: The PointNet model used here was originally trained on the ModelNet10 dataset and has around 85% accuracy. However, this project focuses on demonstrating the feasibility of integrating PointNet with a robotic arm for object manipulation, not on achieving high accuracy. For better performance, refer to the separate repository for a more accurate PointNet model.

## How It Works:
- **Point Cloud Capture**: A camera captures depth images, which are converted into a 3D point cloud.
  
- **Model Classification**: The point cloud is processed by the PointNet model, which predicts the object class.
  
- **Movement and Manipulation**: The robotic arm, controlled using inverse kinematics, moves to the object’s position. After classification, it picks up the object and moves it to a predefined drop-off location.

## Dependencies:
- `pybullet`
- `tensorflow`
- `matplotlib`
- `numpy`

## Running the Project:
1. Ensure you have all dependencies installed.
2. Load the model weights from `CHECKPOINT_PATH` and make sure the URDF files for the robot and objects are correctly referenced.
3. Execute the script to simulate the robotic arm picking up and moving objects.

## Known Issues:
- The gripper issue means the object does not truly "pick up" in a physical sense. The object is simply attached to the gripper at the moment of contact.
  
- Model classification performance may be inconsistent due to limited data for training. Further improvements in model training would require a larger, more diverse dataset.

For reference on improving the model, you can check the other repository where the PointNet model trained on ModelNet10 provides better accuracy.
