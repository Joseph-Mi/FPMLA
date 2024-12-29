import os
import tensorflow as tf
from data_processing import create_data_generators
from model_training import create_quantized_model, train_quantized_model, convert_to_tflite
from model_evaluation import evaluate_model, evaluate_tflite_model
from config import CONFIG

def setup_directories():
    """Create necessary directories for model saving."""
    os.makedirs(os.path.dirname(CONFIG['MODEL_PATH']), exist_ok=True)
    print(f"Created/verified model directory at: {CONFIG['MODEL_PATH']}")

def prepare_data():
    """Create and return data generators."""
    try:
        train_generator, validation_generator = create_data_generators()
        print("Data generators created successfully")
        return train_generator, validation_generator
    except Exception as e:
        print(f"Error creating data generators: {str(e)}")
        raise

def train_and_save_model(train_generator, validation_generator):
    try:
        model = create_quantized_model()
        print("Model created successfully")
        
        history = train_quantized_model(model, train_generator, validation_generator)
        print("Model training completed")
         
        # Save full model - changed to .h5 format
        model_path = os.path.join(CONFIG['MODEL_PATH'], 'final_model.h5')
        model.save(model_path)  # Remove save_format parameter
        print(f"Saved full model to: {model_path}")
        
        return model, history
    except Exception as e:
        print(f"Error in model training/saving: {str(e)}")
        raise

def convert_and_evaluate(model, validation_generator):
    """Convert to TFLite and evaluate both models."""
    try:
        # Convert to TFLite and save to file
        tflite_model = convert_to_tflite(model)
        with open(CONFIG['TFLITE_MODEL_PATH'], 'wb') as f:
            f.write(tflite_model)
        print("Model converted to TFLite successfully")
        
        # Evaluate both models
        keras_accuracy = evaluate_model(model, validation_generator)
        print(f"Keras Model Accuracy: {keras_accuracy['accuracy']:.4f}")
        
        tflite_accuracy = evaluate_tflite_model(
            CONFIG['TFLITE_MODEL_PATH'],  # Pass the path, not the model content
            validation_generator
        )
        print(f"TFLite Model Accuracy: {tflite_accuracy:.4f}")
        
        return keras_accuracy, tflite_accuracy
    except Exception as e:
        print(f"Error in model conversion/evaluation: {str(e)}")
        raise

def analyze_training_history(history):
    """Analyze and plot training metrics."""
    import matplotlib.pyplot as plt
    
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

def main():
    """Main execution function."""
    try:
        print("Starting sign language recognition model training...")
        
        # Setup directories
        setup_directories()
        
        # Prepare data
        train_generator, validation_generator = prepare_data()
        
        # Train model
        model, history = train_and_save_model(train_generator, validation_generator)
        
        # Convert and evaluate
        keras_accuracy, tflite_accuracy = convert_and_evaluate(model, validation_generator)
        
        print("\nTraining completed successfully!")
        print(f"Final Keras Model Accuracy: {keras_accuracy['accuracy']:.4f}")
        print(f"Final TFLite Model Accuracy: {tflite_accuracy:.4f}")
        
    except Exception as e:
        print(f"\nAn error occurred during execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()