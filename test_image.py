from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# for testing 
class_to_word = {
    0: "a / an",
    1: "about",
    2: "above",
    3: "accept / acceptance",
    4: "accord",
    5: "acknowledge",
    6: "acquaint / acquaintance",
    7: "across",
    8: "address",
    9: "advantage",
    10: "advertise",
    11: "after",
    12: "again",
    13: "against",
    14: "agent",
    15: "agree",
    16: "all",
    17: "allow",
    18: "already",
    19: "also",
    20: "altogether",
    21: "always",
    22: "am / more",
    23: "among",
    24: "and / end",
    25: "another",
    26: "answer",
    27: "any",
    28: "appear",
    29: "approximate",
    30: "are / our / hour",
    31: "arrange / arrangement",
    32: "ask",
    33: "attention",
    34: "be / by / but",
    35: "beauty",
    36: "become",
    37: "bed / bad",
    38: "been / bound",
    39: "before",
    40: "beg / big",
    41: "behind",
    42: "belief / believe",
    43: "between",
    44: "bill / built",
    45: "body",
    46: "bring",
    47: "business"
}

def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0  
    return np.expand_dims(img_array, axis=0)

def predict_image(image_path, model_path="steno_model.h5"):
    model = load_model(model_path)
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array) 
    predicted_class = np.argmax(prediction)
    
    predicted_word = class_to_word.get(predicted_class, "Unknown")
    
    """
    print(f"Raw predictions: {prediction}") # for checking
    print(f"Predicted class: {predicted_class}") # for checking what folder/ class 
    """
    return predicted_word

# Testing
image_path = r"C:\Users\jayyy\Desktop\ML\src\test.png"  
predicted_word = predict_image(image_path)
print(f"Predicted word: {predicted_word}")


