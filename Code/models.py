# import os
# from pickle import load
# import numpy as np
# from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, BatchNormalization
# from tensorflow.keras.models import Model
# from tensorflow.keras.utils import plot_model
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# import matplotlib.pyplot as plt

# def define_model_concat(vocab_size, max_length, embedding_matrix, learning_rate, weight_decay, feature_dim):
#     # feature extractor model
#     inputs1 = Input(shape=(feature_dim,))
#     image_feature = Dropout(0.5)(inputs1)
#     image_feature = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(image_feature)
#     image_feature = BatchNormalization()(image_feature)
#     image_feature = Dropout(0.4)(image_feature)
    
#     # sequence model
#     inputs2 = Input(shape=(max_length,))
#     language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
#     language_feature = Dropout(0.5)(language_feature)
#     language_feature = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005)))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)
#     language_feature = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005)))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)
#     language_feature = Bidirectional(LSTM(256, kernel_regularizer=l2(0.0005)))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)
    
#     # decoder model
#     output = concatenate([image_feature, language_feature])
#     output = Dropout(0.4)(output)
#     output = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.4)(output)
#     output = Dense(vocab_size, activation='softmax')(output)
    
#     # tie it together [image, seq] [word]
#     model = Model(inputs=[inputs1, inputs2], outputs=output)
    
#     # create an optimizer with learning rate and weight decay
#     optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
#     # summarize model
#     print(model.summary())
#     # plot_model(model, to_file='model_concat.png', show_shapes=True)
    
#     return model

# def define_model_lstm(vocab_size, max_length, embedding_matrix, learning_rate, weight_decay, feature_dim):
#     # feature extractor model
#     inputs1 = Input(shape=(feature_dim,))
#     image_feature = Dropout(0.5)(inputs1)
#     image_feature = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(image_feature)
#     image_feature = BatchNormalization()(image_feature)
#     image_feature = Dropout(0.4)(image_feature)

#     inputs2 = Input(shape=(max_length,))
#     language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
#     language_feature = Dropout(0.5)(language_feature)
#     language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)
#     language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)
#     language_feature = LSTM(256, kernel_regularizer=l2(0.0005))(language_feature)
#     language_feature = BatchNormalization()(language_feature)
#     language_feature = Dropout(0.4)(language_feature)

#     output = concatenate([image_feature, language_feature])
#     output = Dropout(0.4)(output)
#     output = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(output)
#     output = BatchNormalization()(output)
#     output = Dropout(0.4)(output)
#     output = Dense(vocab_size, activation='softmax')(output)

#     model = Model(inputs=[inputs1, inputs2], outputs=output)
#     optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
#     model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

#     print(model.summary())
    
#     # inputs1 = Input(shape=(feature_dim,))
#     # image_feature = Dropout(0.5)(inputs1)
#     # image_feature = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(image_feature)
#     # image_feature = BatchNormalization()(image_feature)
#     # image_feature = Dropout(0.4)(image_feature)
    
#     # # sequence model
#     # inputs2 = Input(shape=(max_length,))
#     # language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
#     # language_feature = Dropout(0.5)(language_feature)
#     # language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
#     # language_feature = BatchNormalization()(language_feature)
#     # language_feature = Dropout(0.4)(language_feature)
#     # language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
#     # language_feature = BatchNormalization()(language_feature)
#     # language_feature = Dropout(0.4)(language_feature)
#     # language_feature = LSTM(256, kernel_regularizer=l2(0.0005))(language_feature)
#     # language_feature = BatchNormalization()(language_feature)
#     # language_feature = Dropout(0.4)(language_feature)
    
#     # # decoder model
#     # output = concatenate([image_feature, language_feature])
#     # output = Dropout(0.4)(output)
#     # output = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(output)
#     # output = BatchNormalization()(output)
#     # output = Dropout(0.4)(output)
#     # output = Dense(vocab_size, activation='softmax')(output)
    
#     # # tie it together [image, seq] [word]
#     # model = Model(inputs=[inputs1, inputs2], outputs=output)
    
#     # # create an optimizer with learning rate and weight decay
#     # optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    
#     # model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    
#     # # summarize model
#     # print(model.summary())
#     # # plot_model(model, to_file='model_lstm.png', show_shapes=True)
    
#     return model

# def load_generated_files(generated_files_dir):
#     with open(os.path.join(generated_files_dir, 'features_vgg16.pkl'), 'rb') as fid:
#         image_features = load(fid)

#     with open(os.path.join(generated_files_dir, 'caption_train_tokenizer.pkl'), 'rb') as fid:
#         caption_train_tokenizer = load(fid)

#     with open(os.path.join(generated_files_dir, 'image_captions_train.pkl'), 'rb') as fid:
#         image_captions_train = load(fid)

#     with open(os.path.join(generated_files_dir, 'image_captions_dev.pkl'), 'rb') as fid:
#         image_captions_dev = load(fid)

#     with open(os.path.join(generated_files_dir, 'embedding_matrix.pkl'), 'rb') as fid:
#         embedding_matrix = load(fid)

#     return image_features, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix

# def train_model(model, data_gen_train, data_gen_dev, steps_per_epoch, epochs, callbacks, model_name, save_dir):
#     history = model.fit(data_gen_train, epochs=epochs, steps_per_epoch=steps_per_epoch,
#                         validation_data=data_gen_dev, validation_steps=steps_per_epoch,
#                         verbose=1, callbacks=callbacks)
    
#     os.makedirs(save_dir, exist_ok=True)
#     model.save(os.path.join(save_dir, f'{model_name}.keras'))
#     return history

# def plot_training_history(history, model_name):
#     train_losses = history.history['loss']
#     val_losses = history.history['val_loss']
#     train_accuracy = history.history['acc']
#     val_accuracy = history.history['val_acc']

#     # Plotting the loss curves
#     plt.figure()
#     plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
#     plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title(f'Training and Validation Loss - {model_name}')
#     plt.legend()
#     plt.show()

#     # Plotting the accuracy curves
#     plt.figure()
#     plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
#     plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title(f'Training and Validation Accuracy - {model_name}')
#     plt.legend()
#     plt.show()

# models.py
import os
from pickle import load
import numpy as np
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM, Bidirectional, concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W_a = Dense(units)
        self.U_a = Dense(units)
        self.V_a = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = self.V_a(tf.nn.tanh(self.W_a(features) + self.U_a(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

def define_model_concat(vocab_size, max_length, embedding_matrix, learning_rate, weight_decay, feature_dim):
    inputs1 = Input(shape=(feature_dim,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(image_feature)
    image_feature = BatchNormalization()(image_feature)
    image_feature = Dropout(0.4)(image_feature)
    
    inputs2 = Input(shape=(max_length,))
    language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
    language_feature = Dropout(0.5)(language_feature)
    language_feature = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005)))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)
    language_feature = Bidirectional(LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005)))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)
    language_feature = Bidirectional(LSTM(256, kernel_regularizer=l2(0.0005)))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)
    
    output = concatenate([image_feature, language_feature])
    output = Dropout(0.4)(output)
    output = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    output = Dense(vocab_size, activation='softmax')(output)
    
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])
    print(model.summary())
    
    return model

def define_model_lstm(vocab_size, max_length, embedding_matrix, learning_rate, weight_decay, feature_dim):
    inputs1 = Input(shape=(feature_dim,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(image_feature)
    image_feature = BatchNormalization()(image_feature)
    image_feature = Dropout(0.4)(image_feature)

    inputs2 = Input(shape=(max_length,))
    language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
    language_feature = Dropout(0.5)(language_feature)
    language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)
    language_feature = LSTM(256, return_sequences=True, kernel_regularizer=l2(0.0005))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)
    language_feature = LSTM(256, kernel_regularizer=l2(0.0005))(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)

    output = concatenate([image_feature, language_feature])
    output = Dropout(0.4)(output)
    output = Dense(256, activation='relu', kernel_regularizer=l2(0.0005))(output)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    output = Dense(vocab_size, activation='softmax')(output)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    print(model.summary())
    return model

def define_attention_model(vocab_size, max_length, embedding_matrix, learning_rate, weight_decay):
    inputs1 = Input(shape=(4096,))
    image_feature = Dropout(0.5)(inputs1)
    image_feature = Dense(256, activation='relu')(image_feature)
    image_feature = BatchNormalization()(image_feature)
    image_feature = Dropout(0.4)(image_feature)

    inputs2 = Input(shape=(max_length,))
    language_feature = Embedding(vocab_size, 50, weights=[embedding_matrix], input_length=max_length, trainable=False)(inputs2)
    language_feature = Dropout(0.5)(language_feature)
    language_feature, hidden_state, cell_state = LSTM(256, return_sequences=True, return_state=True)(language_feature)
    language_feature = BatchNormalization()(language_feature)
    language_feature = Dropout(0.4)(language_feature)

    attention = BahdanauAttention(256)
    context_vector, attention_weights = attention(image_feature, hidden_state)

    output = concatenate([context_vector, hidden_state])
    output = Dropout(0.5)(output)
    output = Dense(256, activation='relu')(output)
    output = BatchNormalization()(output)
    output = Dropout(0.4)(output)
    output = Dense(vocab_size, activation='softmax')(output)

    model = Model(inputs=[inputs1, inputs2], outputs=output)
    optimizer = Adam(learning_rate=learning_rate, weight_decay=weight_decay)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc'])

    print(model.summary())
    return model

def load_generated_files(generated_files_dir):
    with open(os.path.join(generated_files_dir, 'features_vgg16.pkl'), 'rb') as fid:
        image_features = load(fid)

    with open(os.path.join(generated_files_dir, 'caption_train_tokenizer.pkl'), 'rb') as fid:
        caption_train_tokenizer = load(fid)

    with open(os.path.join(generated_files_dir, 'image_captions_train.pkl'), 'rb') as fid:
        image_captions_train = load(fid)

    with open(os.path.join(generated_files_dir, 'image_captions_dev.pkl'), 'rb') as fid:
        image_captions_dev = load(fid)

    with open(os.path.join(generated_files_dir, 'embedding_matrix.pkl'), 'rb') as fid:
        embedding_matrix = load(fid)

    return image_features, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix

def train_model(model, data_gen_train, data_gen_dev, steps_per_epoch, epochs, callbacks, model_name, save_dir):
    history = model.fit(data_gen_train, epochs=epochs, steps_per_epoch=steps_per_epoch,
                        validation_data=data_gen_dev, validation_steps=steps_per_epoch,
                        verbose=1, callbacks=callbacks)
    
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f'{model_name}.keras'))
    return history

def plot_training_history(history, model_name):
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    train_accuracy = history.history['acc']
    val_accuracy = history.history['val_acc']

    # Plotting the loss curves
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {model_name}')
    plt.legend()
    plt.show()

    # Plotting the accuracy curves
    plt.figure()
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Training and Validation Accuracy - {model_name}')
    plt.legend()
    plt.show()

