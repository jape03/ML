import os
from data_preprocessing import get_data_generators
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# dataset
dataset_path = r"C:\Users\jayyy\Desktop\ML\test_dataset_kaggle"

train_data, val_data = get_data_generators(
    dataset_path,
    target_size=(224, 224),
    batch_size=32
)

# Create the model 
model = create_model(num_classes=train_data.num_classes, fine_tune=True)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

checkpoint = ModelCheckpoint(
    "steno_model.h5",
    save_best_only=True,
    monitor="val_accuracy",
    mode="max"
)
early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

# Trainining the model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,  
    callbacks=[checkpoint, early_stopping]
)

print("Training completed! Model saved as steno_model.h5")
