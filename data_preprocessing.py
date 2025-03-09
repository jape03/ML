from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data_generators(dataset_path, target_size=(224, 224), batch_size=32):

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=30,  # Augmentations to simulate real-world variations
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )

    # No augmentation for validation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2  # Same split as training
    )

    # Training data generator
    train_data = train_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Validation data generator
    val_data = val_datagen.flow_from_directory(
        dataset_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data
