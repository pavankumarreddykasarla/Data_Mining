# import numpy as np
# import pickle
# from tensorflow.keras.applications import InceptionV3, VGG16
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
# from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
# from tensorflow.keras.models import Model, load_model
# from numpy import argmax
# import matplotlib.pyplot as plt
# from PIL import Image

# class FeatureExtractor:
#     def __init__(self, model_name='inception'):
#         self.model_name = model_name
#         self.model = self._load_model()
#         self.feature_extract_pred_model = self._create_feature_extraction_model()
#         self.tokenizer = None
#         self.pred_model = None
#         self.max_length = 33

#     def _load_model(self):
#         if self.model_name == 'inception':
#             return InceptionV3(weights='imagenet')
#         elif self.model_name == 'vgg':
#             return VGG16(weights='imagenet')
#         else:
#             raise ValueError(f"Unsupported model name: {self.model_name}")

#     def _create_feature_extraction_model(self):
#         if self.model_name == 'inception':
#             return Model(inputs=self.model.input, outputs=self.model.get_layer('avg_pool').output)
#         elif self.model_name == 'vgg':
#             return Model(inputs=self.model.input, outputs=self.model.get_layer('fc2').output)

#     def summarize_model(self):
#         self.feature_extract_pred_model.summary()

#     def extract_feature(self, file_name):
#         img = load_img(file_name, target_size=(299, 299) if self.model_name == 'inception' else (224, 224))
#         x = img_to_array(img)
#         x = np.expand_dims(x, axis=0)
#         x = preprocess_inception(x) if self.model_name == 'inception' else preprocess_vgg(x)
#         features = self.feature_extract_pred_model.predict(x)
#         return features

#     def load_tokenizer(self, tokenizer_path):
#         with open(tokenizer_path, 'rb') as file:
#             self.tokenizer = pickle.load(file)

#     def load_prediction_model(self, model_path):
#         self.pred_model = load_model(model_path)

# def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
#     in_text = '<START>'
#     caption_text = list()
#     for i in range(max_length):
#         sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
#         sequence = pad_sequences([sequence], maxlen=max_length)
#         model_softMax_output = pred_model.predict([photo, sequence], verbose=0)
#         word_index = argmax(model_softMax_output)
#         word = caption_train_tokenizer.index_word.get(word_index, None)
#         if word is None:
#             break
#         in_text += ' ' + word
#         if word != 'end':
#             caption_text.append(word)
#         if word == 'end':
#             break
#     return ' '.join(caption_text)

# def flatten(lst):
#     return sum(([x] if not isinstance(x, list) else flatten(x) for x in lst), [])

# def generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width):
#     sequence = caption_train_tokenizer.texts_to_sequences(['<START>'])[0]
#     sequence = pad_sequences([sequence], maxlen=max_length)
#     model_softMax_output = np.squeeze(pred_model.predict([photo, sequence], verbose=0))
#     most_likely_seq = np.argsort(model_softMax_output)[-beam_width:]
#     most_likely_prob = np.log(model_softMax_output[most_likely_seq])

#     most_likely_cap = [[] for _ in range(beam_width)]
#     for j in range(beam_width):
#         most_likely_cap[j] = [[caption_train_tokenizer.index_word[most_likely_seq[j]]]]

#     for i in range(max_length):
#         temp_prob = np.zeros((beam_width, vocab_size))
#         for j in range(beam_width):
#             if most_likely_cap[j][-1] != ['end']:
#                 num_words = len(most_likely_cap[j])
#                 sequence = caption_train_tokenizer.texts_to_sequences(most_likely_cap[j])
#                 sequence = pad_sequences(np.transpose(sequence), maxlen=max_length)
#                 model_softMax_output = pred_model.predict([photo, sequence], verbose=0)
#                 temp_prob[j,] = (1 / num_words) * (most_likely_prob[j] * (num_words - 1) + np.log(model_softMax_output))
#             else:
#                 temp_prob[j,] = most_likely_prob[j] + np.zeros(vocab_size) - np.inf
#                 temp_prob[j, 0] = most_likely_prob[j]

#         x_idx, y_idx = np.unravel_index(temp_prob.flatten().argsort()[-beam_width:], temp_prob.shape)

#         most_likely_cap_temp = [[] for _ in range(beam_width)]
#         for j in range(beam_width):
#             most_likely_prob[j] = temp_prob[x_idx[j], y_idx[j]]
#             most_likely_cap_temp[j] = most_likely_cap[x_idx[j]].copy()
#             if most_likely_cap_temp[j][-1] != ['end']:
#                 most_likely_cap_temp[j].append([caption_train_tokenizer.index_word[y_idx[j]]])

#         most_likely_cap = most_likely_cap_temp.copy()

#         finished = all(cap[-1] == ['end'] for cap in most_likely_cap_temp)

#         if finished:
#             break

#     final_caption = [' '.join(flatten(cap[0:-1])) for cap in most_likely_cap]

#     return final_caption, most_likely_prob

import numpy as np
import pickle
from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inception
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.models import Model, load_model
from numpy import argmax
import matplotlib.pyplot as plt
from PIL import Image

class FeatureExtractor:
    def __init__(self, model_name='inception', features_path=None, tokenizer_path=None):
        self.model_name = model_name
        self.model = self._load_model()
        self.feature_extract_pred_model = self._create_feature_extraction_model()
        self.tokenizer = None
        self.pred_model = None
        self.max_length = 33
        if features_path and tokenizer_path:
            self.load_features_and_tokenizer(features_path, tokenizer_path)

    def _load_model(self):
        if self.model_name == 'inception':
            return InceptionV3(weights='imagenet')
        elif self.model_name == 'vgg':
            return VGG16(weights='imagenet')
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

    def _create_feature_extraction_model(self):
        if self.model_name == 'inception':
            return Model(inputs=self.model.input, outputs=self.model.get_layer('avg_pool').output)
        elif self.model_name == 'vgg':
            return Model(inputs=self.model.input, outputs=self.model.get_layer('fc2').output)

    def summarize_model(self):
        self.feature_extract_pred_model.summary()

    def extract_feature(self, file_name):
        img = load_img(file_name, target_size=(299, 299) if self.model_name == 'inception' else (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_inception(x) if self.model_name == 'inception' else preprocess_vgg(x)
        features = self.feature_extract_pred_model.predict(x)
        return features

    def load_tokenizer(self, tokenizer_path):
        with open(tokenizer_path, 'rb') as file:
            self.tokenizer = pickle.load(file)

    def load_prediction_model(self, model_path):
        self.pred_model = load_model(model_path)

    def load_features_and_tokenizer(self, features_path, tokenizer_path):
        with open(features_path, 'rb') as fid:
            self.image_features = pickle.load(fid)
        self.load_tokenizer(tokenizer_path)

def generate_caption(pred_model, caption_train_tokenizer, photo, max_length):
    in_text = '<START>'
    caption_text = list()
    for i in range(max_length):
        sequence = caption_train_tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        model_softMax_output = pred_model.predict([photo, sequence], verbose=0)
        word_index = argmax(model_softMax_output)
        word = caption_train_tokenizer.index_word.get(word_index, None)
        if word is None:
            break
        in_text += ' ' + word
        if word != 'end':
            caption_text.append(word)
        if word == 'end':
            break
    return ' '.join(caption_text)

def flatten(lst):
    return sum(([x] if not isinstance(x, list) else flatten(x) for x in lst), [])

def generate_caption_beam(pred_model, caption_train_tokenizer, photo, max_length, vocab_size, beam_width):
    sequence = caption_train_tokenizer.texts_to_sequences(['<START>'])[0]
    sequence = pad_sequences([sequence], maxlen=max_length)
    model_softMax_output = np.squeeze(pred_model.predict([photo, sequence], verbose=0))
    most_likely_seq = np.argsort(model_softMax_output)[-beam_width:]
    most_likely_prob = np.log(model_softMax_output[most_likely_seq])

    most_likely_cap = [[] for _ in range(beam_width)]
    for j in range(beam_width):
        most_likely_cap[j] = [[caption_train_tokenizer.index_word[most_likely_seq[j]]]]

    for i in range(max_length):
        temp_prob = np.zeros((beam_width, vocab_size))
        for j in range(beam_width):
            if most_likely_cap[j][-1] != ['end']:
                num_words = len(most_likely_cap[j])
                sequence = caption_train_tokenizer.texts_to_sequences(most_likely_cap[j])
                sequence = pad_sequences(np.transpose(sequence), maxlen=max_length)
                model_softMax_output = pred_model.predict([photo, sequence], verbose=0)
                temp_prob[j,] = (1 / num_words) * (most_likely_prob[j] * (num_words - 1) + np.log(model_softMax_output))
            else:
                temp_prob[j,] = most_likely_prob[j] + np.zeros(vocab_size) - np.inf
                temp_prob[j, 0] = most_likely_prob[j]

        x_idx, y_idx = np.unravel_index(temp_prob.flatten().argsort()[-beam_width:], temp_prob.shape)

        most_likely_cap_temp = [[] for _ in range(beam_width)]
        for j in range(beam_width):
            most_likely_prob[j] = temp_prob[x_idx[j], y_idx[j]]
            most_likely_cap_temp[j] = most_likely_cap[x_idx[j]].copy()
            if most_likely_cap_temp[j][-1] != ['end']:
                most_likely_cap_temp[j].append([caption_train_tokenizer.index_word[y_idx[j]]])

        most_likely_cap = most_likely_cap_temp.copy()

        finished = all(cap[-1] == ['end'] for cap in most_likely_cap_temp)

        if finished:
            break

    final_caption = [' '.join(flatten(cap[0:-1])) for cap in most_likely_cap]

    return final_caption, most_likely_prob
