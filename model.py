from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

def create_model(num_classes, fine_tune=False):
    # MobileNetV2 
    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )

    if fine_tune:
        base_model.trainable = True
        for layer in base_model.layers[:-50]:  
            layer.trainable = False
    else:
        base_model.trainable = False

    # Build 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(), 
        layers.Reshape((1, 1280)),  
        layers.Bidirectional(layers.LSTM(256, return_sequences=False)),  # BiLSTM layer
        layers.Dropout(0.5), 
        layers.Dense(256, activation='relu'),  
        layers.Dropout(0.5),  
        layers.Dense(num_classes, activation='softmax') 
    ])
    return model
