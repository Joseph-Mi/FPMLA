import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    'IMAGE_SIZE': (64, 64),
    'BATCH_SIZE': 16,  # Increased from 8
    'EPOCHS': 180,
    'LEARNING_RATE': 0.0005,  # slower learning rate
    'VALIDATION_SPLIT': 0.1,
    'QUANTIZATION_AWARE': True,
    'TARGET_DEVICE': 'DE1_SOC',
    'FPGA_PREPROCESSING': False,  # Disabled for initial training

    # Data paths
    'TRAIN_PATH': os.path.join(ROOT_DIR, 'data', 'train'),
    'VALIDATION_PATH': os.path.join(ROOT_DIR, 'data', 'validation'),
    'TEST_PATH': os.path.join(ROOT_DIR, 'data', 'test'),
    # Model paths
    # Model paths
    'MODEL_PATH': os.path.join(ROOT_DIR, 'models', 'saved_models'),  # Directory for all models
    'MODEL_FILE': os.path.join(ROOT_DIR, 'models', 'saved_models', 'model.h5'),  # Full model file
    'BEST_MODEL_FILE': os.path.join(ROOT_DIR, 'models', 'saved_models', 'best_model.h5'),  # Best checkpoint
    'TFLITE_MODEL_PATH': os.path.join(ROOT_DIR, 'models', 'saved_models', 'model.tflite')  # TFLite model
}