import os
import numpy as np
import h5py
from tqdm import tqdm
import trimesh 

def load_off(file_path):
    """
    Load vertices and faces from an .off file.
    Automatically triangulate non-triangular faces using trimesh.
    """
    mesh = trimesh.load_mesh(file_path)
    vertices = np.array(mesh.vertices)  # Extract vertices
    faces = np.array(mesh.faces)       # Extract faces (automatically triangulated by trimesh)
    return vertices, faces

def sample_points(vertices, faces, num_points=1024):
    """
    Sample points uniformly from the mesh surface.
    """
    face_areas = np.zeros(len(faces))
    for i, face in enumerate(faces):
        v1, v2, v3 = vertices[face]
        face_areas[i] = np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2
    face_probs = face_areas / np.sum(face_areas)
    chosen_faces = np.random.choice(len(faces), size=num_points, p=face_probs)

    sampled_points = []
    for idx in chosen_faces:
        v1, v2, v3 = vertices[faces[idx]]
        r1, r2 = np.random.rand(2)
        sqrt_r1 = np.sqrt(r1)
        point = (1 - sqrt_r1) * v1 + sqrt_r1 * (1 - r2) * v2 + sqrt_r1 * r2 * v3
        sampled_points.append(point)
    return np.array(sampled_points)

def normalize_point_cloud(points):
    """
    Center the cloud and scale it to fit inside a unit sphere.
    """
    centroid = np.mean(points, axis=0)
    points -= centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / scale

def preprocess_shapes(root_dir, output_file, num_points=1024):
    """
    Preprocess cubes, spheres, triangles into HDF5 with normalized point clouds.
    """
    categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    label_map = {cat: i for i, cat in enumerate(categories)}  # cubes: 0, spheres: 1, triangles: 2
    
    all_train_points = []
    all_train_labels = []
    all_test_points = []
    all_test_labels = []

    for cat in categories:
        for split in ["train", "test"]:
            split_dir = os.path.join(root_dir, cat, split)
            files = [f for f in os.listdir(split_dir) if f.endswith('.off')]
            for file_name in tqdm(files, desc=f"{split.upper()} {cat}"):
                file_path = os.path.join(split_dir, file_name)
                vertices, faces = load_off(file_path)
                
                if faces.shape[0] == 0:
                    print(f"Skipping {file_path} because it has no triangular faces.")
                    continue
                
                points = sample_points(vertices, faces, num_points)
                points = normalize_point_cloud(points)

                if split == "train":
                    all_train_points.append(points)
                    all_train_labels.append(label_map[cat])
                else:
                    all_test_points.append(points)
                    all_test_labels.append(label_map[cat])

    # Save to HDF5
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('train_points', data=np.array(all_train_points))
        f.create_dataset('train_labels', data=np.array(all_train_labels))
        f.create_dataset('test_points', data=np.array(all_test_points))
        f.create_dataset('test_labels', data=np.array(all_test_labels))

if __name__ == "__main__":
    root_dir = "C://Users//iammd//Desktop//pointnet-robotic_arm//data"
    output_file = "robotic_shapes_preprocessed.h5"
    preprocess_shapes(root_dir, output_file)