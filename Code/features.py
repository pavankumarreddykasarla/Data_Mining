import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
from tensorflow.keras.applications.vgg16 import VGG16 as VGG16_Model
from tensorflow.keras.applications.inception_v3 import InceptionV3 as InceptionV3_Model
from pickle import dump

class FeatureExtractor:
    def __init__(self, dataset_path, features_directory):
        self.dataset_path = dataset_path
        self.features_directory = features_directory
        self.features = dict()

    def extract_features(self, model_name='vgg'):
        if model_name == 'vgg':
            model = VGG16_Model(weights='imagenet', include_top=False, pooling='avg')
            preprocess_input = preprocess_input_vgg
            img_size = (224, 224)
            features_filename = os.path.join(self.features_directory, 'features_vgg.pkl')
        elif model_name == 'inception':
            model = InceptionV3_Model(weights='imagenet', include_top=False, pooling='avg')
            preprocess_input = preprocess_input_inception
            img_size = (299, 299)
            features_filename = os.path.join(self.features_directory, 'features_inception.pkl')
        else:
            print("Unsupported model_name. Use 'vgg' or 'inception'.")
            return

        for file in os.listdir(self.dataset_path):
            img_path = os.path.join(self.dataset_path, file)
            img = load_img(img_path, target_size=img_size)  
            x = img_to_array(img) 
            x = np.expand_dims(x, axis=0)  # Expand to include batch dimension at the beginning
            x = preprocess_input(x)  # Preprocess input according to the model
            features = model.predict(x)
            
            name_id = file.split('.')[0]  # Take the file name and use as id in dict
            self.features[name_id] = features

        self.save_features(features_filename)

    def save_features(self, output_path):
        with open(output_path, 'wb') as f:
            dump(self.features, f)  # Cannot use JSON because ndarray is not JSON serializable

if __name__ == "__main__":
    # Fetch data from the "Data" folder in the parent directory
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
    features_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Features')

    if not os.path.exists(features_directory):
        os.makedirs(features_directory)

    extractor = FeatureExtractor(dataset_path, features_directory)

    # Extract features using VGG16
    extractor.extract_features(model_name='vgg')

    # Extract features using InceptionV3
    extractor.extract_features(model_name='inception')
