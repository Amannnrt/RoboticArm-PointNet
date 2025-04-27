import pybullet as p
import pybullet_data
import time

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

cube_start_position = [0,0,0.5]
cube_start_orientation = p.getQuaternionFromEuler([0,0,0])
cube_id = p.loadURDF("cube.urdf",cube_start_position,cube_start_orientation)

while True:
    p.stepSimulation()
    time.sleep(1.0/240)