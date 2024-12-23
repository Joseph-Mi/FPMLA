import tensorflow as tf
import numpy as np

def evaluate_model(model, test_generator):
    """
    Evaluate the Keras model on the test generator.
    Returns a dictionary with loss and accuracy.
    """
    results = model.evaluate(test_generator)
    return {
        'loss': results[0],
        'accuracy': results[1]
    }

def evaluate_tflite_model(tflite_model_path, validation_generator):
    """Evaluate TFLite model accuracy."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Reset the generator to the beginning
    validation_generator.reset()
    
    correct_predictions = 0
    total_predictions = 0
    
    for i in range(len(validation_generator)):
        batch_images, batch_labels = validation_generator[i]
        
        # Process each image in the batch individually
        for image, label in zip(batch_images, batch_labels):
            # Add batch dimension and ensure correct shape
            input_data = np.expand_dims(image, axis=0)
            
            # Quantize input if necessary
            if input_details[0]['dtype'] == np.int8:
                input_scale, input_zero_point = input_details[0]['quantization']
                input_data = input_data / input_scale + input_zero_point
                input_data = input_data.astype(np.int8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Dequantize output if necessary
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            predicted_label = np.argmax(output_data)
            true_label = np.argmax(label)
            
            if predicted_label == true_label:
                correct_predictions += 1
            total_predictions += 1
    
    return correct_predictions / total_predictions