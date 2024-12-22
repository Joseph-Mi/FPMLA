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

def evaluate_tflite_model(tflite_path, test_generator):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    accuracy = 0
    total = 0
    
    for images, labels in test_generator:
        # Quantize input if using int8
        if input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = input_details[0]['quantization']
            images = images / input_scale + input_zero_point
            images = images.astype(np.int8)
            
        interpreter.set_tensor(input_details[0]['index'], images)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output if using int8
        if output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = output_details[0]['quantization']
            predictions = (predictions.astype(np.float32) - output_zero_point) * output_scale
            
        accuracy += tf.reduce_sum(tf.cast(tf.argmax(predictions, axis=1) == tf.argmax(labels, axis=1), tf.float32))
        total += len(labels)
    
    return float(accuracy / total)