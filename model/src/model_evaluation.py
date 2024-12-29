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

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from config import CONFIG


def evaluate_tflite_model(tflite_model_path, test_generator):
    """Evaluates the converted TFLite model with improved preprocessing."""
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    correct_predictions = 0
    total_predictions = 0
    
    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    
    for i in range(len(test_generator)):
        batch_images, batch_labels = test_generator[i]
        for j in range(len(batch_images)):
            # Preprocess input
            input_data = batch_images[j:j+1]
            
            # Improved quantization process
            input_data = input_data / input_scale + input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Improved dequantization process
            if output_details[0]['dtype'] == np.int8:
                output_scale, output_zero_point = output_details[0]['quantization']
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            predicted_class = np.argmax(output_data[0])
            true_class = np.argmax(batch_labels[j])
            
            if predicted_class == true_class:
                correct_predictions += 1
            total_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy


# def create_quantized_model():
#     """Creates a model with Quantization-Aware Training (QAT)."""
#     with tfmot.quantization.keras.quantize_scope():
#         inputs = tf.keras.Input(shape=(64, 64, 1))
        
#         # First Conv Block with Residual
#         x = quantized_conv_block(inputs, 32, is_first_layer=True)
#         x = quantized_residual_block(x, 32)
        
#         # Second Conv Block with Residual
#         x = quantized_conv_block(x, 64)
#         x = quantized_residual_block(x, 64)
        
#         # Third Conv Block with Residual
#         x = quantized_conv_block(x, 128)
#         x = quantized_residual_block(x, 128)
        
#         # Global Average Pooling
#         x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
#         # Dense Layers with quantization
#         x = quantized_dense_block(x, 256)
        
#         # Output layer
#         outputs = tf.keras.layers.Dense(24)(x)
#         outputs = tf.keras.layers.Activation('softmax')(outputs)
        
#         model = tf.keras.Model(inputs, outputs)
    
#     # Apply quantization to the entire model
#     quantized_model = tfmot.quantization.keras.quantize_model(model)
    
#     # Configure learning rate schedule
#     initial_learning_rate = CONFIG['LEARNING_RATE']
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate,
#         decay_steps=100,
#         decay_rate=0.9
#     )
    
#     quantized_model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#         loss='categorical_crossentropy',
#         metrics=['accuracy']
#     )
    
#     return quantized_model

# def quantized_conv_block(x, filters, is_first_layer=False):
#     """Creates a quantized convolutional block."""
#     if is_first_layer:
#         conv = tf.keras.layers.Conv2D(
#             filters, (3, 3), 
#             padding='same',
#             input_shape=(64, 64, 1)
#         )(x)
#     else:
#         conv = tf.keras.layers.Conv2D(
#             filters, (3, 3), 
#             padding='same'
#         )(x)
        
#     x = tf.keras.layers.BatchNormalization()(conv)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    
#     return x

# def quantized_residual_block(x, filters):
#     """Creates a quantized residual block."""
#     residual = x
    
#     x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
    
#     x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
    
#     if residual.shape[-1] != filters:
#         residual = tf.keras.layers.Conv2D(
#             filters, (1, 1),
#             kernel_initializer='he_normal'
#         )(residual)
    
#     x = tf.keras.layers.Add()([x, residual])
#     x = tf.keras.layers.Activation('relu')(x)
    
#     return x

# def quantized_dense_block(x, units):
#     """Creates a quantized dense block."""
#     x = tf.keras.layers.Dense(units)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation('relu')(x)
#     x = tf.keras.layers.Dropout(0.6)(x)
#     return x

# def train_quantized_model(model, train_generator, validation_generator):
#     """Trains the quantized model."""
#     callbacks = [
#         tf.keras.callbacks.ModelCheckpoint(
#             CONFIG['MODEL_PATH'] + 'best_quantized_model.h5',
#             save_best_only=True,
#             monitor='val_accuracy',
#             mode='max'
#         ),
#         tf.keras.callbacks.ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.2,
#             patience=6,
#             min_lr=1e-6
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=10,
#             restore_best_weights=True
#         )
#     ]
    
#     history = model.fit(
#         train_generator,
#         validation_data=validation_generator,
#         epochs=CONFIG['EPOCHS'],
#         callbacks=callbacks
#     )
    
#     return history, model

# def representative_dataset_gen():
#     """Generates representative dataset for quantization."""
#     validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#         rescale=1./255
#     )
#     validation_generator = validation_datagen.flow_from_directory(
#         CONFIG['VALIDATION_PATH'],
#         target_size=(64, 64),
#         batch_size=1,
#         class_mode='categorical',
#         color_mode='grayscale',
#         shuffle=False
#     )
    
#     for _ in range(100):  # Calibrate with representative samples
#         image, _ = next(validation_generator)
#         yield [image.astype('float32')]

# def convert_to_tflite(model, save_path=None):
#     """Converts the trained QAT model to TFLite format with int8 quantization."""
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
#     # Enable optimizations
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
#     # Enforce full integer quantization
#     converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#     converter.inference_input_type = tf.int8
#     converter.inference_output_type = tf.int8
    
#     # Set representative dataset for calibration
#     def representative_dataset_gen():
#         """Generate representative dataset with robust statistics."""
#         validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1./255,
#             # Add slight augmentation to make calibration more robust
#             rotation_range=2,
#             width_shift_range=0.02,
#             height_shift_range=0.02
#         )
        
#         validation_generator = validation_datagen.flow_from_directory(
#             CONFIG['VALIDATION_PATH'],
#             target_size=(64, 64),
#             batch_size=1,
#             class_mode='categorical',
#             color_mode='grayscale',
#             shuffle=True  # Enable shuffling for better representation
#         )
        
#         # Increase number of calibration samples
#         num_calibration_steps = 200  # Doubled from previous version
        
#         # Use moving averages for more stable statistics
#         values_list = []
        
#         for _ in range(num_calibration_steps):
#             image, _ = next(validation_generator)
#             image = image.astype('float32')
#             values_list.append(image)
            
#             # Remove outliers by using percentiles instead of min/max
#             if len(values_list) >= 10:
#                 stacked_values = np.concatenate(values_list)
#                 # Use 1st and 99th percentiles instead of min/max
#                 min_val = np.percentile(stacked_values, 1)
#                 max_val = np.percentile(stacked_values, 99)
                
#                 yield [np.clip(image, min_val, max_val)]
    
#     converter.representative_dataset = representative_dataset_gen
    
#     # Additional optimizations
#     converter.target_spec.supported_types = [tf.int8]
#     converter.experimental_new_converter = True
    
#     # Let TensorFlow handle the exact quantization parameters
#     # Remove manual setting of quantization parameters
    
#     # Convert the model
#     tflite_model = converter.convert()
    
#     # Save the model if path is provided
#     if save_path:
#         with open(save_path, 'wb') as f:
#             f.write(tflite_model)
    
#     return tflite_model