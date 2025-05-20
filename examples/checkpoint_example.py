"""
Example showing how to save and load model checkpoints with ParamLake.
"""

import os
import numpy as np
import tensorflow as tf
import paramlake
from paramlake import save_checkpoint, load_checkpoint, list_checkpoints

# Create a directory for storing data
os.makedirs("checkpoint_example_data", exist_ok=True)

# Create a simple model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compile the model with SGD optimizer
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Generate some dummy data
def generate_data():
    x_train = np.random.random((1000, 784))
    y_train = np.random.randint(0, 10, (1000,))
    x_test = np.random.random((200, 784))
    y_test = np.random.randint(0, 10, (200,))
    return (x_train, y_train), (x_test, y_test)

# Create a ParamLake storage manager
def create_storage():
    config = {
        "output_path": "checkpoint_example_data/model_data.zarr",
        "run_id": "checkpoint_example",
        "capture_frequency": 1,
        "capture_gradients": True,
        "capture_weights": True,
        "capture_optimizer_state": True,
    }
    
    # Create and return a storage manager
    from paramlake.storage.factory import create_storage_manager
    from paramlake.utils.config import ParamLakeConfig
    
    config_obj = ParamLakeConfig(config)
    return create_storage_manager(config_obj)

# Main example workflow
def main():
    print("Creating model and data...")
    model = create_model()
    (x_train, y_train), (x_test, y_test) = generate_data()
    
    # Create storage manager
    storage = create_storage()
    
    # Train for a few epochs
    print("\nTraining model (initial phase)...")
    model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test), verbose=1)
    
    # Save a checkpoint after initial training
    print("\nSaving checkpoint after initial training...")
    checkpoint_id = save_checkpoint(
        model=model,
        storage_manager=storage,
        step=3,
        name="initial_training",
        description="Checkpoint after 3 epochs"
    )
    print(f"Saved checkpoint with ID: {checkpoint_id}")
    
    # Continue training for a few more epochs
    print("\nContinuing training...")
    model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test), verbose=1, initial_epoch=3)
    
    # Save another checkpoint
    print("\nSaving checkpoint after additional training...")
    checkpoint_id_2 = save_checkpoint(
        model=model,
        storage_manager=storage,
        step=5,
        name="additional_training",
        description="Checkpoint after 5 epochs"
    )
    print(f"Saved checkpoint with ID: {checkpoint_id_2}")
    
    # List all checkpoints
    print("\nListing all checkpoints:")
    checkpoints = list_checkpoints(storage)
    for i, checkpoint in enumerate(checkpoints):
        print(f"Checkpoint {i+1}:")
        print(f"  ID: {checkpoint.get('id', 'N/A')}")
        print(f"  Name: {checkpoint.get('name', 'N/A')}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
        print(f"  Description: {checkpoint.get('description', 'N/A')}")
        print(f"  Timestamp: {checkpoint.get('timestamp', 'N/A')}")
    
    # Create a new model instance
    print("\nCreating new model instance...")
    new_model = create_model()
    
    # Evaluate the new model before loading checkpoint
    print("\nEvaluating new model before loading checkpoint:")
    loss, acc = new_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy before loading checkpoint: {acc:.4f}")
    
    # Load the first checkpoint into the new model
    print("\nLoading first checkpoint into new model...")
    metadata = load_checkpoint(
        model=new_model,
        storage_manager=storage,
        checkpoint_id=checkpoint_id,
        include_optimizer=True,
        recompile=True
    )
    print(f"Loaded checkpoint metadata: {metadata}")
    
    # Evaluate the new model after loading checkpoint
    print("\nEvaluating model after loading checkpoint:")
    loss, acc = new_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after loading checkpoint: {acc:.4f}")
    
    # Continue training from the loaded checkpoint
    print("\nContinuing training from loaded checkpoint...")
    new_model.fit(x_train, y_train, epochs=2, batch_size=32, validation_data=(x_test, y_test), verbose=1, initial_epoch=3)
    
    # Demonstrate load by step number
    print("\nCreating another model and loading checkpoint by step number...")
    another_model = create_model()
    
    # Load by step
    metadata = load_checkpoint(
        model=another_model,
        storage_manager=storage,
        step=3,  # Load checkpoint from step 3
        include_optimizer=True
    )
    
    # Evaluate
    loss, acc = another_model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy after loading checkpoint by step: {acc:.4f}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main() 