import tensorflow as tf
from config import CONFIG

def create_model():
    inputs = tf.keras.Input(shape=(64, 64, 1))
    
    # First Conv Block with Residual
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 32)
    
    # Second Conv Block with Residual
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 64)
    
    # Third Conv Block with Residual
    x = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = residual_block(x, 128)
    
    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense Layers with increased regularization
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.6)(x)  # Increased dropout
    
    outputs = tf.keras.layers.Dense(24, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # Compile with learning rate schedule
    initial_learning_rate = CONFIG['LEARNING_RATE']
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=100, decay_rate=0.9
    )
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def residual_block(x, filters):
    residual = x
    
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Match dimensions for residual connection
    if residual.shape[-1] != filters:
        residual = tf.keras.layers.Conv2D(filters, (1, 1))(residual)
        
    x = tf.keras.layers.Add()([x, residual])
    x = tf.keras.layers.Activation('relu')(x)
    
    return x

def convert_to_tflite(model, enable_hard_swish = True, enable_dynamic_range = False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # base optimization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Configure quantization
    converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.target_spec.supported_types = [tf.int8]
    converter.representative_dataset = representative_dataset_gen
    
    # Enable full integer quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.optimize_transforms = ["hard_swish_transform"]  # Optional optimization
    
    #optional
    if enable_hard_swish:
            converter.optimize_transforms = ["hard_swish_transform"]
            
    if enable_dynamic_range:
        converter.optimizations.append(tf.lite.Optimize.OPTIMIZE_FOR_SIZE)

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
            monitor='val_accuracy',  # judge on accuracy
            mode='max'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=6,
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
