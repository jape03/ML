from tensorflow.keras.models import load_model
from data_preprocessing import get_data_generators

dataset_path = r"C:\Users\jayyy\Desktop\ML\test_dataset_kaggle"

# Load model
model = load_model("steno_model.h5")
_, val_data = get_data_generators(dataset_path)

# Evaluate model
loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.2f}")
