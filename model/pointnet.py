import tensorflow as tf
from tensorflow.keras import layers, Model

def t_net(inputs, num_features):
    x = layers.Conv1D(32, 1, activation='relu')(inputs)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)

    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)

    x = layers.Dense(num_features * num_features, activation=None)(x)
    x = layers.Reshape((num_features, num_features))(x)

    def add_identity(t):
        identity = tf.eye(num_features, batch_shape=[tf.shape(t)[0]])
        return t + identity

    x = layers.Lambda(add_identity, name=f"t_net_identity_add_{num_features}")(x)
    return x

def pointnet_model(num_classes, num_points=1024):
    inputs = layers.Input(shape=(num_points, 3))

    # Transform the input points using T-Net
    input_transform = t_net(inputs, num_features=3)
    transformed_inputs = layers.Lambda(lambda x: tf.matmul(x[0], x[1]), name="input_transform_matmul")([inputs, input_transform])

    # Apply convolutions to the transformed input points
    x = layers.Conv1D(32, 1, activation='relu')(transformed_inputs)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)

    # Apply feature transformation using T-Net
    feature_transform = t_net(x, num_features=64)
    x = layers.Lambda(lambda x: tf.matmul(x[0], x[1]), name="feature_transform_matmul")([x, feature_transform])

    # More convolutions
    x = layers.Conv1D(64, 1, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Conv1D(128, 1, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)

    # Global feature extraction
    global_feature = layers.GlobalMaxPooling1D()(x)

    # Fully connected layers
    x = layers.Dense(256, activation='relu')(global_feature)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    return model

# //////////////////////////////////////////////////////////////////////////

# import tensorflow as tf
# from tensorflow.keras import layers, Model

# def t_net(inputs, num_features):
#     # Add L2 regularization to all layers
#     reg = tf.keras.regularizers.l2(0.001)
    
#     x = layers.Conv1D(32, 1, activation='relu', kernel_regularizer=reg)(inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv1D(64, 1, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv1D(128, 1, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)

#     x = layers.GlobalMaxPooling1D()(x)

#     x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.3)(x)
#     x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.3)(x)

#     x = layers.Dense(num_features * num_features)(x)
#     x = layers.Reshape((num_features, num_features))(x)

#     def add_identity(t):
#         identity = tf.eye(num_features, batch_shape=[tf.shape(t)[0]])
#         return t + identity

#     return layers.Lambda(add_identity)(x)

# def pointnet_model(num_classes, num_points=1024):
#     inputs = layers.Input(shape=(num_points, 3))
    
#     # Input transformation
#     input_transform = t_net(inputs, 3)
#     transformed_inputs = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([inputs, input_transform])

#     # Feature extraction with regularization
#     reg = tf.keras.regularizers.l2(0.001)
#     x = layers.Conv1D(32, 1, activation='relu', kernel_regularizer=reg)(transformed_inputs)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.2)(x)
    
#     x = layers.Conv1D(64, 1, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.2)(x)

#     # Feature transformation
#     feature_transform = t_net(x, 64)
#     x = layers.Lambda(lambda x: tf.matmul(x[0], x[1]))([x, feature_transform])

#     x = layers.Conv1D(64, 1, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Conv1D(128, 1, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)

#     global_feature = layers.GlobalMaxPooling1D()(x)

#     # Classification head with heavy regularization
#     x = layers.Dense(256, activation='relu', kernel_regularizer=reg)(global_feature)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Dense(128, activation='relu', kernel_regularizer=reg)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Dropout(0.4)(x)
    
#     outputs = layers.Dense(num_classes, activation='softmax')(x)

#     return Model(inputs=inputs, outputs=outputs)