import tensorflow as tf
from config import CONFIG

def create_model():
    model = tf.keras.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Fourth Conv Block
        tf.keras.layers.Conv2D(256, (3, 3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Global Average Pooling instead of Flatten
        tf.keras.layers.GlobalAveragePooling2D(),
        
        # Dense Layers
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(24, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['LEARNING_RATE']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def convert_to_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Configure quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.representative_dataset = representative_dataset_gen
    
    # Enable full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.optimize_transforms = ["hard_swish_transform"]  # Optional optimization
    
    tflite_model = converter.convert()
    
    with open(CONFIG['TFLITE_MODEL_PATH'], 'wb') as f:
        f.write(tflite_model)
    
    return tflite_model

# Add this function to provide calibration data
def representative_dataset_gen():
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )
    validation_generator = validation_datagen.flow_from_directory(
        CONFIG['VALIDATION_PATH'],
        target_size=CONFIG['IMAGE_SIZE'],
        batch_size=1,
        class_mode='categorical',
        color_mode='grayscale',
        shuffle=False
    )
    
    for i in range(220):  # Calibrate with 100 images ////// NUMBER MIGHT NEED TO CHANGE
        image, _ = next(validation_generator)
        yield [image]

def train_model(model, train_generator, validation_generator):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            CONFIG['MODEL_PATH'] + 'best_model.h5',
            save_best_only=True,
            monitor='val_accuracy'  # judge on accuracy
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
            )
    ]
    
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=CONFIG['EPOCHS'],
        callbacks=callbacks
    )
    
    return history
