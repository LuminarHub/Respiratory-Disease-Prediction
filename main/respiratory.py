import tensorflow as tf
import numpy as np
from PIL import Image
import os

class_labels = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

model_path= os.path.join(os.getcwd(),'Respiratory.h5')

# filename='D:/Projects/Parkingson_s/Respiratory/Test Data/Bacterial Pneumonia.jpeg'

def load_and_prep_image(filename, img_shape=128):
    img = Image.open(filename)
    img = img.convert('RGB')
    img = img.resize((img_shape, img_shape))
    img = np.array(img) / 255.0 
    img = np.expand_dims(img, axis=0)
    return img



def predicted_class(class_labels,model_path,test_image):
    saved_model = tf.keras.models.load_model(model_path)
    prediction=saved_model.predict(test_image)
    predicted_class = np.argmax(prediction, axis=-1)
    predicted_class_label = class_labels[predicted_class[0]]
    print(f"Predicted class: {predicted_class_label}")
    return predicted_class_label

# predicted_class(class_labels,model_path,test_image)