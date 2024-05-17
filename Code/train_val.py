# # # # main.py
# # # from utils import FlickrDataset, ImageFeatureExtractor, DataPreparation
# # # from embeddings import EmbeddingMatrixBuilder
# # # from models import define_model_concat, load_generated_files, train_model, plot_training_history
# # # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# # # import os
# # # from pickle import load

# # # def main():
# # #     # Load the Flickr dataset
# # #     flickr_dataset = FlickrDataset()
# # #     flickr_dataset.load_and_process_captions('Flickr.token.txt')
# # #     flickr_dataset.save_data()

# # #     print(f"Size of data: {len(flickr_dataset.image_captions)}")
# # #     print(f"Size of training data: {len(flickr_dataset.image_captions_train)}")
# # #     print(f"Size of dev data: {len(flickr_dataset.image_captions_dev)}")
# # #     print(f"Size of test data: {len(flickr_dataset.image_captions_test)}")
# # #     print(f"Size of unused data: {len(flickr_dataset.image_captions_other)}")
# # #     print(f"Maximum image caption length: {flickr_dataset.max_imageCap_len}")

# # #     # Initialize the feature extractor
# # #     feature_extractor = ImageFeatureExtractor()
# # #     feature_extractor.summarize_model()

# # #     # Load the tokenizer
# # #     flickr_dataset.load_tokenizer()
# # #     word_index = flickr_dataset.caption_train_tokenizer.word_index

# # #     # Path to the embedding matrix file
# # #     embedding_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'embedding_matrix.pkl'))

# # #     # Check if the embedding matrix file exists
# # #     if not os.path.exists(embedding_output_path):
# # #         print("Embedding matrix file not found. Creating and saving the embedding matrix.")
# # #         embedding_builder = EmbeddingMatrixBuilder()
# # #         embedding_matrix = embedding_builder.build_embedding_matrix(word_index)
# # #         embedding_builder.save_embedding_matrix(embedding_matrix, embedding_output_path)
# # #     else:
# # #         print("Embedding matrix file found. Loading the embedding matrix.")
# # #         with open(embedding_output_path, 'rb') as fid:
# # #             embedding_matrix = load(fid)

# # #     # Load generated files
# # #     generated_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
# # #     image_features, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix = load_generated_files(generated_files_dir)

# # #     # Define model parameters
# # #     caption_max_length = flickr_dataset.max_imageCap_len
# # #     vocab_size = len(word_index) + 1
# # #     learning_rate = 0.001
# # #     weight_decay = 1e-4

# # #     # Define model
# # #     VGG_model = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
# # #     Inception_model = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay,feature_dim=2048)
# # #     # Prepare data for model
# # #     data_preparation = DataPreparation(generated_files_dir)
# # #     batch_size = 64  # Example batch size

# # #     # Create generators
# # #     data_gen_train = data_preparation.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
# # #     data_gen_dev = data_preparation.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
# # #     steps_train = len(image_captions_train) // batch_size
# # #     steps_dev = len(image_captions_dev) // batch_size
# # #     epochs = 40

# # #     # Use EarlyStopping to monitor validation loss and stop training early if no improvement is seen
# # #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# # #     # Use ReduceLROnPlateau to reduce learning rate when a metric has stopped improving
# # #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# # #     # Train the model
# # #     history_VGG = train_model(VGG_model, data_gen_train, data_gen_dev, steps_train, epochs, [early_stopping, reduce_lr])
# # #     history_Inception = train_model(Inception_model, data_gen_train, data_gen_dev, steps_train, epochs, [early_stopping, reduce_lr])

# # #     # Plot training history
# # #     plot_training_history(history_VGG)
# # #     plot_training_history(history_Inception)
# # # if __name__ == "__main__":
# # #     main()

# # # main.py
# # # main.py
# # from utils import FlickrDataset, ImageFeatureExtractor, DataPreparation
# # from embeddings import EmbeddingMatrixBuilder
# # from models import define_model_concat, load_generated_files, train_model, plot_training_history
# # from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# # import os
# # from pickle import load

# # def main():
# #     # Load the Flickr dataset
# #     flickr_dataset = FlickrDataset()
# #     flickr_dataset.load_and_process_captions('Flickr.token.txt')
# #     flickr_dataset.save_data()

# #     print(f"Size of data: {len(flickr_dataset.image_captions)}")
# #     print(f"Size of training data: {len(flickr_dataset.image_captions_train)}")
# #     print(f"Size of dev data: {len(flickr_dataset.image_captions_dev)}")
# #     print(f"Size of test data: {len(flickr_dataset.image_captions_test)}")
# #     print(f"Size of unused data: {len(flickr_dataset.image_captions_other)}")
# #     print(f"Maximum image caption length: {flickr_dataset.max_imageCap_len}")

# #     # Load the tokenizer
# #     flickr_dataset.load_tokenizer()
# #     word_index = flickr_dataset.caption_train_tokenizer.word_index

# #     # Path to the embedding matrix file
# #     embedding_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'embedding_matrix.pkl'))

# #     # Check if the embedding matrix file exists
# #     if not os.path.exists(embedding_output_path):
# #         print("Embedding matrix file not found. Creating and saving the embedding matrix.")
# #         embedding_builder = EmbeddingMatrixBuilder()
# #         embedding_matrix = embedding_builder.build_embedding_matrix(word_index)
# #         embedding_builder.save_embedding_matrix(embedding_matrix, embedding_output_path)
# #     else:
# #         print("Embedding matrix file found. Loading the embedding matrix.")
# #         with open(embedding_output_path, 'rb') as fid:
# #             embedding_matrix = load(fid)

# #     # Load generated files for VGG16
# #     generated_files_dir_vgg = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
# #     image_features_vgg, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix = load_generated_files(generated_files_dir_vgg)

# #     # Load generated files for InceptionV3
# #     generated_files_dir_inception = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
# #     with open(os.path.join(generated_files_dir_inception, 'features_inception.pkl'), 'rb') as fid:
# #         image_features_inception = load(fid)

# #     # Define model parameters
# #     caption_max_length = flickr_dataset.max_imageCap_len
# #     vocab_size = len(word_index) + 1
# #     learning_rate = 0.001
# #     weight_decay = 1e-4

# #     # Define models
# #     model_vgg = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
# #     model_inception = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=2048)

# #     # Prepare data for models
# #     data_preparation_vgg = DataPreparation(generated_files_dir_vgg, feature_type='vgg')
# #     data_preparation_inception = DataPreparation(generated_files_dir_inception, feature_type='inception')

# #     batch_size = 64  # Example batch size

# #     # Create generators for VGG16
# #     data_gen_train_vgg = data_preparation_vgg.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
# #     data_gen_dev_vgg = data_preparation_vgg.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
# #     # Create generators for InceptionV3
# #     data_gen_train_inception = data_preparation_inception.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
# #     data_gen_dev_inception = data_preparation_inception.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
# #     steps_train = len(image_captions_train) // batch_size
# #     steps_dev = len(image_captions_dev) // batch_size
# #     epochs = 2

# #     # Use EarlyStopping to monitor validation loss and stop training early if no improvement is seen
# #     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# #     # Use ReduceLROnPlateau to reduce learning rate when a metric has stopped improving
# #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

# #     # Train the models
# #     save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
# #     print("VGG_LSTM is training")
# #     history_vgg = train_model(model_vgg, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_vgg', save_dir)
# #     print("Inception_LSTM is training")
# #     history_inception = train_model(model_inception, data_gen_train_inception, data_gen_dev_inception, steps_train, epochs, [early_stopping, reduce_lr], 'model_inception', save_dir)

# #     # Plot training history
# #     plot_training_history(history_vgg)
# #     plot_training_history(history_inception)

# # if __name__ == "__main__":
# #     main()

# # main.py

# from utils import FlickrDataset, ImageFeatureExtractor, DataPreparation
# from embeddings import EmbeddingMatrixBuilder
# from models import define_model_concat, define_model_lstm, load_generated_files, train_model, plot_training_history
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# import os
# from pickle import load

# def main():
#     # Load the Flickr dataset
#     flickr_dataset = FlickrDataset()
#     flickr_dataset.load_and_process_captions('Flickr.token.txt')
#     flickr_dataset.save_data()

#     print(f"Size of data: {len(flickr_dataset.image_captions)}")
#     print(f"Size of training data: {len(flickr_dataset.image_captions_train)}")
#     print(f"Size of dev data: {len(flickr_dataset.image_captions_dev)}")
#     print(f"Size of test data: {len(flickr_dataset.image_captions_test)}")
#     print(f"Size of unused data: {len(flickr_dataset.image_captions_other)}")
#     print(f"Maximum image caption length: {flickr_dataset.max_imageCap_len}")

#     # Load the tokenizer
#     flickr_dataset.load_tokenizer()
#     word_index = flickr_dataset.caption_train_tokenizer.word_index

#     # Path to the embedding matrix file
#     embedding_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'embedding_matrix.pkl'))

#     # Check if the embedding matrix file exists
#     if not os.path.exists(embedding_output_path):
#         print("Embedding matrix file not found. Creating and saving the embedding matrix.")
#         embedding_builder = EmbeddingMatrixBuilder()
#         embedding_matrix = embedding_builder.build_embedding_matrix(word_index)
#         embedding_builder.save_embedding_matrix(embedding_matrix, embedding_output_path)
#     else:
#         print("Embedding matrix file found. Loading the embedding matrix.")
#         with open(embedding_output_path, 'rb') as fid:
#             embedding_matrix = load(fid)

#     # Load generated files
#     generated_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
#     image_features_vgg, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix = load_generated_files(generated_files_dir)

#     # Load Inception features
#     with open(os.path.join(generated_files_dir, 'features_inception.pkl'), 'rb') as fid:
#         image_features_inception = load(fid)

#     # Define model parameters
#     caption_max_length = flickr_dataset.max_imageCap_len
#     vocab_size = len(word_index) + 1
#     learning_rate = 0.001
#     weight_decay = 1e-4

#     # Define models
#     model_vgg_bilstm = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
#     model_inception_bilstm = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=2048)
#     model_vgg_lstm = define_model_lstm(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
#     model_inception_lstm = define_model_lstm(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=2048)

#     # Prepare data for models
#     data_preparation_vgg = DataPreparation(generated_files_dir, feature_type='vgg')
#     data_preparation_inception = DataPreparation(generated_files_dir, feature_type='inception')

#     batch_size = 64  # Example batch size

#     # Create generators for VGG16
#     data_gen_train_vgg = data_preparation_vgg.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
#     data_gen_dev_vgg = data_preparation_vgg.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
#     # Create generators for InceptionV3
#     data_gen_train_inception = data_preparation_inception.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
#     data_gen_dev_inception = data_preparation_inception.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
#     steps_train = len(image_captions_train) // batch_size
#     steps_dev = len(image_captions_dev) // batch_size
#     epochs = 2

#     # Use EarlyStopping to monitor validation loss and stop training early if no improvement is seen
#     early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#     # Use ReduceLROnPlateau to reduce learning rate when a metric has stopped improving
#     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

#     # Train the models
#     save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
#     print("Training VGG BiLSTM model...")
#     history_vgg_bilstm = train_model(model_vgg_bilstm, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_vgg_bilstm', save_dir)
#     print("Training Inception BiLSTM model...")
#     history_inception_bilstm = train_model(model_inception_bilstm, data_gen_train_inception, data_gen_dev_inception, steps_train, epochs, [early_stopping, reduce_lr], 'model_inception_bilstm', save_dir)
#     print("Training VGG LSTM model...")
#     history_vgg_lstm = train_model(model_vgg_lstm, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_vgg_lstm', save_dir)
#     print("Training Inception LSTM model...")
#     history_inception_lstm = train_model(model_inception_lstm, data_gen_train_inception, data_gen_dev_inception, steps_train, epochs, [early_stopping, reduce_lr], 'model_inception_lstm', save_dir)

#     # Plot training history
#     plot_training_history(history_vgg_bilstm)
#     plot_training_history(history_inception_bilstm)
#     plot_training_history(history_vgg_lstm)
#     plot_training_history(history_inception_lstm)

# if __name__ == "__main__":
#     main()

# main.py

from utils import FlickrDataset, ImageFeatureExtractor, DataPreparation
from embeddings import EmbeddingMatrixBuilder
from models import define_model_concat, define_model_lstm, define_attention_model, load_generated_files, train_model, plot_training_history
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from pickle import load

def main():
    # Load the Flickr dataset
    flickr_dataset = FlickrDataset()
    flickr_dataset.load_and_process_captions('Flickr.token.txt')
    flickr_dataset.save_data()

    print(f"Size of data: {len(flickr_dataset.image_captions)}")
    print(f"Size of training data: {len(flickr_dataset.image_captions_train)}")
    print(f"Size of dev data: {len(flickr_dataset.image_captions_dev)}")
    print(f"Size of test data: {len(flickr_dataset.image_captions_test)}")
    print(f"Size of unused data: {len(flickr_dataset.image_captions_other)}")
    print(f"Maximum image caption length: {flickr_dataset.max_imageCap_len}")

    # Load the tokenizer
    flickr_dataset.load_tokenizer()
    word_index = flickr_dataset.caption_train_tokenizer.word_index

    # Path to the embedding matrix file
    embedding_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'embedding_matrix.pkl'))

    # Check if the embedding matrix file exists
    if not os.path.exists(embedding_output_path):
        print("Embedding matrix file not found. Creating and saving the embedding matrix.")
        embedding_builder = EmbeddingMatrixBuilder()
        embedding_matrix = embedding_builder.build_embedding_matrix(word_index)
        embedding_builder.save_embedding_matrix(embedding_matrix, embedding_output_path)
    else:
        print("Embedding matrix file found. Loading the embedding matrix.")
        with open(embedding_output_path, 'rb') as fid:
            embedding_matrix = load(fid)

    # Load generated files
    generated_files_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))
    image_features_vgg, caption_train_tokenizer, image_captions_train, image_captions_dev, embedding_matrix = load_generated_files(generated_files_dir)

    # Load Inception features
    with open(os.path.join(generated_files_dir, 'features_inception.pkl'), 'rb') as fid:
        image_features_inception = load(fid)

    # Define model parameters
    caption_max_length = flickr_dataset.max_imageCap_len
    vocab_size = len(word_index) + 1
    learning_rate = 0.001
    weight_decay = 1e-4

    # Define models
    model_vgg_bilstm = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
    model_inception_bilstm = define_model_concat(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=2048)
    model_vgg_lstm = define_model_lstm(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=4096)
    model_inception_lstm = define_model_lstm(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay, feature_dim=2048)
    model_attention = define_attention_model(vocab_size, caption_max_length, embedding_matrix, learning_rate, weight_decay)

    # Prepare data for models
    data_preparation_vgg = DataPreparation(generated_files_dir, feature_type='vgg')
    data_preparation_inception = DataPreparation(generated_files_dir, feature_type='inception')

    batch_size = 64  # Example batch size

    # Create generators for VGG16
    data_gen_train_vgg = data_preparation_vgg.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    data_gen_dev_vgg = data_preparation_vgg.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
    # Create generators for InceptionV3
    data_gen_train_inception = data_preparation_inception.data_generator(image_captions_train, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    data_gen_dev_inception = data_preparation_inception.data_generator(image_captions_dev, caption_train_tokenizer, caption_max_length, batch_size, vocab_size)
    
    steps_train = len(image_captions_train) // batch_size
    steps_dev = len(image_captions_dev) // batch_size
    epochs = 40

    # Use EarlyStopping to monitor validation loss and stop training early if no improvement is seen
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Use ReduceLROnPlateau to reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)

    # Train the models
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files'))

    print("Training VGG BiLSTM model...")
    history_vgg_bilstm = train_model(model_vgg_bilstm, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_vgg_bilstm', save_dir)
    plot_training_history(history_vgg_bilstm, 'VGG BiLSTM')

    print("Training Inception BiLSTM model...")
    history_inception_bilstm = train_model(model_inception_bilstm, data_gen_train_inception, data_gen_dev_inception, steps_train, epochs, [early_stopping, reduce_lr], 'model_inception_bilstm', save_dir)
    plot_training_history(history_inception_bilstm, 'Inception BiLSTM')

    print("Training VGG LSTM model...")
    history_vgg_lstm = train_model(model_vgg_lstm, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_vgg_lstm', save_dir)
    plot_training_history(history_vgg_lstm, 'VGG LSTM')

    print("Training Inception LSTM model...")
    history_inception_lstm = train_model(model_inception_lstm, data_gen_train_inception, data_gen_dev_inception, steps_train, epochs, [early_stopping, reduce_lr], 'model_inception_lstm', save_dir)
    plot_training_history(history_inception_lstm, 'Inception LSTM')

    print("Training Attention model...")
    history_attention = train_model(model_attention, data_gen_train_vgg, data_gen_dev_vgg, steps_train, epochs, [early_stopping, reduce_lr], 'model_attention', save_dir)
    plot_training_history(history_attention, 'Attention')

if __name__ == "__main__":
    main()
