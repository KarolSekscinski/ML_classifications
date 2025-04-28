# TensorFlow / Keras imports for MLP
import tensorflow as tf
from tensorflow.keras.layers import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam  # Common optimizer




# --- Keras MLP Model Definition ---
def create_mlp_model(hidden_layer_sizes=(64, 32), activation='relu',
                     dropout_rate=0.3, learning_rate=0.001,
                     meta=None):  # meta contains input/output shapes provided by scikeras
    """
    Creates a compiled Keras Sequential MLP model.
    Designed to be passed to KerasClassifier.

    Args:
        hidden_layer_sizes (tuple): Number of units in each hidden layer.
        activation (str): Activation function for hidden layers.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Learning rate for the Adam optimizer.
        meta (dict): Dictionary containing metadata like input shape, target shape, etc.
                     Automatically passed by KerasClassifier.

    Returns:
        tf.keras.models.Sequential: Compiled Keras model.
    """
    # Get input shape from metadata provided by KerasClassifier
    # This is crucial because the shape depends on the preprocessing output (e.g., after OHE)
    if meta is None:
        raise ValueError("Metadata (meta) is required to determine input shape.")
    n_features_in_ = meta.get("n_features_in_", None)
    target_info = meta.get("target_info_", {})  # Get target info
    n_classes_ = target_info.get("n_classes_", 1)  # Default to 1 for binary regression-like output

    if n_features_in_ is None:
        raise ValueError("Could not determine input shape ('n_features_in_') from metadata.")

    model = Sequential(name="MLP_Classifier")
    model.add(Input(shape=(n_features_in_,), name="input_layer"))  # Define input layer explicitly

    # Add hidden layers
    for units in hidden_layer_sizes:
        model.add(Dense(units, activation=activation))
        model.add(Dropout(dropout_rate))  # Add dropout for regularization

    # Output layer
    # Use 'sigmoid' for binary classification (outputting a single probability)
    # n_classes_ should ideally be 1 for binary classification with sigmoid
    if n_classes_ <= 2:  # Binary classification case
        output_units = 1
        output_activation = 'sigmoid'
        loss_function = 'binary_crossentropy'
    else:  # Multiclass classification case (adapt if needed)
        output_units = n_classes_
        output_activation = 'softmax'
        loss_function = 'sparse_categorical_crossentropy'  # If y is integer labels
        # loss_function = 'categorical_crossentropy' # If y is one-hot encoded

    model.add(Dense(output_units, activation=output_activation, name="output_layer"))

    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])  # Add AUC metric

    logger.info(
        f"Created Keras MLP model with input shape=({n_features_in_},), output units={output_units}, loss={loss_function}")
    # model.summary(print_fn=logger.info) # Log model summary if needed

    return model
