import numpy as np
import tensorflow as tf
from utils.dataloader import get_datasets
from model.pointnet import pointnet_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Paths
CHECKPOINT_PATH = "checkpoints/best_model.h5"
HDF5_FILE_PATH = "robotic_shapes_preprocessed.h5"

# Hyperparameters
BATCH_SIZE = 16
NUM_CLASSES = 3

# Specify the target test categories
TARGET_CATEGORIES = [0, 1, 2]  # Change this to the desired categories


def check_label_distribution(dataset):
    """
    Check the distribution of labels in the dataset.
    """
    label_counts = {}
    for _, batch_labels in dataset:
        for label in batch_labels.numpy():
            label_counts[label] = label_counts.get(label, 0) + 1

    print("Label Distribution:")
    for label, count in sorted(label_counts.items()):
        print(f"Category {label}: {count} samples")


def filter_dataset_by_categories(dataset, target_categories):
    """
    Filters the dataset to include only samples from the target categories.
    """
    filtered_points = []
    filtered_labels = []

    for batch_points, batch_labels in dataset:
        mask = np.isin(batch_labels.numpy(), target_categories)
        filtered_points.append(batch_points.numpy()[mask])
        filtered_labels.append(batch_labels.numpy()[mask])

    # Concatenate all filtered batches
    filtered_points = np.concatenate(filtered_points, axis=0)
    filtered_labels = np.concatenate(filtered_labels, axis=0)

    # Create a new TensorFlow dataset
    filtered_dataset = tf.data.Dataset.from_tensor_slices((filtered_points, filtered_labels))
    filtered_dataset = filtered_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return filtered_dataset


def visualize_point_cloud(points, title="Point Cloud"):
    """
    Visualize a single point cloud in 3D.
    """
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def visualize_predictions_by_category(model, filtered_dataset, target_categories):
    """
    Visualize one prediction per target category.
    """
    samples = {cat: None for cat in target_categories}

    for batch_points, batch_labels in filtered_dataset:
        predictions = model.predict(batch_points)
        predicted_classes = np.argmax(predictions, axis=1)

        for i in range(len(batch_labels)):
            true_label = batch_labels[i].numpy()
            if true_label in target_categories and samples[true_label] is None:
                samples[true_label] = (batch_points[i].numpy(), true_label, predicted_classes[i])

        if all(value is not None for value in samples.values()):
            break

    for cat, result in samples.items():
        if result is not None:
            points, true_label, predicted_label = result
            visualize_point_cloud(
                points,
                title=f"Category {cat} — True: {true_label}, Predicted: {predicted_label}"
            )
        else:
            print(f"No sample found for category {cat}")


def test():
    # Rebuild the model architecture
    print("Rebuilding the model architecture...")
    model = pointnet_model(num_classes=NUM_CLASSES)

    # Load the pre-trained weights
    print("Loading the pre-trained weights...")
    model.load_weights(CHECKPOINT_PATH)

    # Recompile the model
    print("Recompiling the model...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Load datasets
    _, test_dataset = get_datasets(HDF5_FILE_PATH, BATCH_SIZE)

    # Check the label distribution
    print("Checking label distribution...")
    check_label_distribution(test_dataset)

    # Filter the test dataset by the target categories
    print(f"Filtering test dataset for categories {TARGET_CATEGORIES}...")
    filtered_test_dataset = filter_dataset_by_categories(test_dataset, TARGET_CATEGORIES)

    # Evaluate the model on the filtered dataset
    print("Evaluating the model...")
    loss, accuracy = model.evaluate(filtered_test_dataset)
    print(f"Test Loss for Categories {TARGET_CATEGORIES}: {loss:.4f}")
    print(f"Test Accuracy for Categories {TARGET_CATEGORIES}: {accuracy:.4f}")

    # Visualize one sample per category
    print("Visualizing one sample per category...")
    visualize_predictions_by_category(model, filtered_test_dataset, TARGET_CATEGORIES)


if __name__ == "__main__":
    test()
