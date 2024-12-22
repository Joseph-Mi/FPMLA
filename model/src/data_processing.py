import tensorflow as tf
from config import CONFIG
import os

def preprocess_for_fpga(image):
    # First ensure we're working with float32
    image = tf.cast(image, tf.float32)
    
    # Scale to [-128, 127] range
    image = tf.multiply(image, 255.0)  # Scale to [0, 255]
    image = image - 128.0  # Center around 0 to get [-128, 127]
    
    # Clip values to ensure they're in the valid range
    image = tf.clip_by_value(image, -128.0, 127.0)
    
    # Convert to int8 after all floating point operations are done
    image = tf.cast(image, tf.int8)
    
    # Validate range using properly typed tensors
    max_val = tf.cast(127, tf.int8)
    min_val = tf.cast(-128, tf.int8)
    
    # Add validation checks
    tf.debugging.assert_less_equal(
        tf.cast(tf.reduce_max(image), tf.int8),
        max_val,
        message="Pixel values exceed 127"
    )
    tf.debugging.assert_greater_equal(
        tf.cast(tf.reduce_min(image), tf.int8),
        min_val,
        message="Pixel values below -128"
    )
    
    # Validate shape
    tf.debugging.assert_equal(tf.shape(image)[-1], 1, message="Image is not grayscale")
    
    return image

def create_data_generators():
    # Create a preprocessing function that combines rescaling and FPGA preprocessing
    def combined_preprocessing(image):
        # Normalize to [0,1] range first
        image = image / 255.0
        # Then apply FPGA preprocessing if enabled in config
        if CONFIG.get('FPGA_PREPROCESSING', False):
            image = preprocess_for_fpga(image)
        return image

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=combined_preprocessing,
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8,1.2], 
        shear_range=0.1,           
        fill_mode='nearest',        
        zoom_range=0.2,
        horizontal_flip=False
    )

    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=combined_preprocessing
    )

    train_generator = train_datagen.flow_from_directory(
        CONFIG['TRAIN_PATH'],
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        color_mode='grayscale'
    )

    validation_generator = validation_datagen.flow_from_directory(
        CONFIG['VALIDATION_PATH'],
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=CONFIG['BATCH_SIZE'],
        class_mode='categorical',
        color_mode='grayscale'
    )

    return train_generator, validation_generator