import os
import tensorflow as tf
from model.pointnet import pointnet_model  # This should work if pointnet.py is in models folder
from utils.dataloader import get_datasets  # Ensure this is correct and the dataloader is in utils folder

# Hyperparameters
BATCH_SIZE = 8
NUM_CLASSES = 3  # Update based on your dataset (cubes: 0, spheres: 1, triangles: 2)
EPOCHS = 25
LEARNING_RATE = 0.0001

# Paths
HDF5_FILE_PATH = "robotic_shapes_preprocessed.h5"  # Update with the correct dataset path
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def train():
    # Step 1: Load datasets
    train_dataset, test_dataset = get_datasets(HDF5_FILE_PATH, BATCH_SIZE)

    # Step 2: Define the PointNet model
    model = pointnet_model(num_classes=NUM_CLASSES)

    # Step 3: Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Step 4: Set up callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(CHECKPOINT_DIR, "best_model.h5"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    )
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=LOG_DIR,
        histogram_freq=1
    )

    # Step 5: Train the model
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=EPOCHS,
        callbacks=[checkpoint_callback, tensorboard_callback]
    )

    # Step 6: Save the final model
    model.save(os.path.join(CHECKPOINT_DIR, "final_model.h5"))

if __name__ == "__main__":
    train()


# ///////////////////////////////////////////////////////////////////

# import os
# import numpy as np
# import tensorflow as tf
# from model.pointnet import pointnet_model
# from utils.dataloader import get_datasets

# # Hyperparameters
# BATCH_SIZE = 16
# NUM_CLASSES = 3
# EPOCHS = 10
# LEARNING_RATE = 0.001
# AUGMENTATION = True

# # Paths
# HDF5_FILE_PATH = "robotic_shapes_preprocessed.h5"
# CHECKPOINT_DIR = "checkpoints/"
# LOG_DIR = "logs/"

# os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# os.makedirs(LOG_DIR, exist_ok=True)

# def random_rotation(point_cloud):
#     angle = tf.random.uniform(shape=(), minval=0, maxval=2 * np.pi)
#     cos_a = tf.math.cos(angle)
#     sin_a = tf.math.sin(angle)
#     rotation_matrix = tf.reshape([
#         cos_a,  sin_a, 0.0,
#         -sin_a, cos_a, 0.0,
#         0.0,    0.0,   1.0
#     ], shape=(3, 3))
#     return tf.matmul(point_cloud, rotation_matrix)

# def augment_data(point_cloud, label):
#     point_cloud = random_rotation(point_cloud)
#     jitter = tf.random.normal(
#         shape=tf.shape(point_cloud),
#         mean=0.0,
#         stddev=0.02
#     )
#     point_cloud += jitter
#     scale = tf.random.uniform(
#         shape=[1], 
#         minval=0.8, 
#         maxval=1.2
#     )
#     point_cloud *= scale
#     return point_cloud, label

# def smoothed_sparse_crossentropy(y_true, y_pred):
#     num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
#     smoothing = tf.constant(0.1, dtype=tf.float32)
    
#     y_true = tf.cast(y_true, tf.int32)
#     one_hot = tf.one_hot(y_true, depth=tf.cast(num_classes, tf.int32))
#     one_hot = tf.squeeze(one_hot, axis=1)
#     one_hot = one_hot * (1.0 - smoothing) + (smoothing / num_classes)
    
#     return tf.keras.losses.categorical_crossentropy(
#         tf.cast(one_hot, tf.float32),
#         y_pred, 
#         from_logits=False
#     )

# def train():
#     train_dataset, test_dataset = get_datasets(HDF5_FILE_PATH, BATCH_SIZE)

#     if AUGMENTATION:
#         train_dataset = train_dataset.map(
#             augment_data,
#             num_parallel_calls=tf.data.AUTOTUNE
#         ).prefetch(tf.data.AUTOTUNE)

#     model = pointnet_model(NUM_CLASSES)
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
#         loss=smoothed_sparse_crossentropy,
#         metrics=['accuracy']
#     )

#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             os.path.join(CHECKPOINT_DIR, "best_model.h5"),
#             monitor="val_accuracy",
#             save_best_only=True,
#             verbose=1
#         ),
#         tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_accuracy',
#             patience=15,
#             restore_best_weights=True
#         )
#     ]

#     history = model.fit(
#         train_dataset,
#         validation_data=test_dataset,
#         epochs=EPOCHS,
#         callbacks=callbacks,
#         verbose=1
#     )

#     model.save(os.path.join(CHECKPOINT_DIR, "final_model.h5"))

# if __name__ == "__main__":
#     train()