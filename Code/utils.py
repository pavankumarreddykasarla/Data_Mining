
# utils.py
import os
import string
import numpy as np
from pickle import dump, load
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_vgg
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inception
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from numpy import argmax
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import Orthogonal

class FlickrDataset:
    def __init__(self, output_dir='../Generated_files'):
        # Navigate to the Data directory to access the Flickr_text folder
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'Flickr_text'))
        self.base_dir = base_dir
        self.output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), output_dir))
        self.training_set = self._load_data_set_ids('Flickr.trainImages.txt')
        self.dev_set = self._load_data_set_ids('Flickr.devImages.txt')
        self.test_set = self._load_data_set_ids('Flickr.testImages.txt')
        self.translator = str.maketrans("", "", string.punctuation)
        self.image_captions = dict()
        self.image_captions_train = dict()
        self.image_captions_dev = dict()
        self.image_captions_test = dict()
        self.image_captions_other = dict()
        self.corpus = ['<START>', '<END>', '<UNK>']
        self.max_imageCap_len = 0
        self.caption_train_tokenizer = None

    def _load_data_set_ids(self, filename):
        filepath = os.path.join(self.base_dir, filename)
        with open(filepath, 'r') as file:
            text = file.read()
        
        dataset = [image_id for image_id in text.split('\n') if len(image_id) > 0]
        return set(dataset)

    def load_and_process_captions(self, caption_file):
        filepath = os.path.join(self.base_dir, caption_file)
        with open(filepath, 'r') as file:
            token_text = file.read()
        
        for line in token_text.split('\n'):
            if len(line) < 2:
                continue
            tokens = line.split(' ')
            image_id, image_cap = tokens[0], tokens[1:]
            image_id = image_id.split('#')[0]
            image_cap = ' '.join(image_cap).lower().translate(self.translator).split(' ')
            image_cap = [w for w in image_cap if w.isalpha() and len(w) > 1]
            image_cap = '<START> ' + ' '.join(image_cap) + ' <END>'
            
            if len(image_cap.split()) > self.max_imageCap_len:
                self.max_imageCap_len = len(image_cap.split())

            if image_id not in self.image_captions:
                self.image_captions[image_id] = list()
            self.image_captions[image_id].append(image_cap)

            if image_id in self.training_set:
                if image_id not in self.image_captions_train:
                    self.image_captions_train[image_id] = list()
                self.image_captions_train[image_id].append(image_cap)
                self.corpus.extend(image_cap.split())
            elif image_id in self.dev_set:
                if image_id not in self.image_captions_dev:
                    self.image_captions_dev[image_id] = list()
                self.image_captions_dev[image_id].append(image_cap)
            elif image_id in self.test_set:
                if image_id not in self.image_captions_test:
                    self.image_captions_test[image_id] = list()
                self.image_captions_test[image_id].append(image_cap)
            else:
                if image_id not in self.image_captions_other:
                    self.image_captions_other[image_id] = list()
                self.image_captions_other[image_id].append(image_cap)

    def save_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        self._save_to_file(os.path.join(self.output_dir, 'image_captions.pkl'), self.image_captions)
        self._save_to_file(os.path.join(self.output_dir, 'image_captions_train.pkl'), self.image_captions_train)
        self._save_to_file(os.path.join(self.output_dir, 'image_captions_dev.pkl'), self.image_captions_dev)
        self._save_to_file(os.path.join(self.output_dir, 'image_captions_test.pkl'), self.image_captions_test)
        self._save_to_file(os.path.join(self.output_dir, 'image_captions_other.pkl'), self.image_captions_other)
        self._save_to_file(os.path.join(self.output_dir, 'corpus.pkl'), self.corpus)
        self._save_to_file(os.path.join(self.output_dir, 'corpus_count.pkl'), Counter(self.corpus))

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.corpus)
        self.caption_train_tokenizer = tokenizer
        self._save_to_file(os.path.join(self.output_dir, 'caption_train_tokenizer.pkl'), tokenizer)

    def load_tokenizer(self):
        tokenizer_path = os.path.join(self.output_dir, 'caption_train_tokenizer.pkl')
        with open(tokenizer_path, 'rb') as file:
            self.caption_train_tokenizer = load(file)
    
    def _save_to_file(self, filepath, data):
        with open(filepath, 'wb') as file:
            dump(data, file)

class ImageFeatureExtractor:
    def __init__(self, model_name='VGG16'):
        if model_name == 'VGG16':
            base_model = VGG16(weights='imagenet')
            self.feature_dim = 4096
            self.preprocess_input = preprocess_input_vgg
        elif model_name == 'InceptionV3':
            base_model = InceptionV3(weights='imagenet')
            self.feature_dim = 2048
            self.preprocess_input = preprocess_input_inception
        else:
            raise ValueError("Unsupported model name")
        
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('avg_pool').output)

    def summarize_model(self):
        self.model.summary()
    
    def extract_features(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224)) if self.feature_dim == 4096 else image.load_img(image_path, target_size=(299, 299))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = self.preprocess_input(img_array)
        
        features = self.model.predict(img_array)
        return features

class DataPreparation:
    def __init__(self, generated_files_directory, feature_type='vgg'):
        self.generated_files_directory = generated_files_directory
        self.feature_type = feature_type
        self.image_features = self._load_features()

    def _load_features(self):
        features_filename = 'features_vgg16.pkl' if self.feature_type == 'vgg' else 'features_inception.pkl'
        features_path = os.path.join(self.generated_files_directory, features_filename)
        try:
            with open(features_path, 'rb') as fid:
                features = load(fid)
            return features
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return {}

    def create_sequences(self, tokenizer, max_length, desc_list, photo, vocab_size):
        X1, X2, y = list(), list(), list()
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            for i in range(1, len(seq)):
                in_seq, out_seq = seq[:i], seq[i]
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                X1.append(photo)
                X2.append(in_seq)
                y.append(out_seq)
        return np.array(X1), np.array(X2), np.array(y)

    def data_generator(self, descriptions, tokenizer, max_length, batch_size, vocab_size):
        while True:
            X1, X2, Y = list(), list(), list()
            for key, desc_list in descriptions.items():
                imageFeature_id = key.split('.')[0]
                photo = self.image_features[imageFeature_id][0]
                in_img, in_seq, out_word = self.create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                X1.extend(in_img)
                X2.extend(in_seq)
                Y.extend(out_word)
                if len(X1) >= batch_size:
                    yield ((np.array(X1), np.array(X2)), np.array(Y))
                    X1, X2, Y = list(), list(), list()

    def get_tf_dataset(self, descriptions, tokenizer, max_length, batch_size, vocab_size):
        def generator():
            for key, desc_list in descriptions.items():
                imageFeature_id = key.split('.')[0]
                photo = self.image_features[imageFeature_id][0]
                X1, X2, Y = self.create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
                for i in range(len(X1)):
                    yield (X1[i], X2[i]), Y[i]

        output_signature = (
            (tf.TensorSpec(shape=(self.feature_dim,), dtype=tf.float32),
             tf.TensorSpec(shape=(max_length,), dtype=tf.int32)),
            tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
        )

        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        dataset = dataset.batch(batch_size)
        return dataset
    
