import os
from beamsearch_utils import FeatureExtractor, generate_caption, generate_caption_beam
import matplotlib.pyplot as plt
from PIL import Image
from pickle import load, dump
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
from rouge import Rouge
import textwrap

def generate_captions_and_scores(model_name='inception', model_variant='lstm', beam_search=False, beam_width=5):
    # Define paths
    features_file_name = 'features_inception.pkl' if model_name == 'inception' else 'features_vgg16.pkl'
    features_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Features', features_file_name))
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'caption_train_tokenizer.pkl'))
    
    if model_name == 'inception':
        if model_variant == 'lstm':
            model_file_name = 'model_inception_lstm.keras'
        else:
            model_file_name = 'model_inception_bilstm.keras'
    elif model_name == 'vgg' and model_variant == 'attention':
        model_file_name = 'model_vgg_bilstm.keras'
    else:
        if model_variant == 'lstm':
            model_file_name = 'model_vgg_lstm.keras'
        else:
            model_file_name = 'model_vgg_bilstm.keras'
    
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', model_file_name))
    image_captions_test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Generated_files', 'image_captions_test.pkl'))
    
    # Create an instance of FeatureExtractor for the specified model
    feature_extractor = FeatureExtractor(model_name=model_name, features_path=features_path, tokenizer_path=tokenizer_path)
    
    # Load the prediction model
    feature_extractor.load_prediction_model(model_path)
    
    # Load test captions
    with open(image_captions_test_path, 'rb') as fid:
        image_captions_test = load(fid)
    
    # Select a single image for testing
    image_fileName = list(image_captions_test.keys())[0]
    reference_captions = image_captions_test[image_fileName]
    
    # Generate captions for the test image using greedy search
    image_fileName_feature = image_fileName.split('.')[0]
    photo = feature_extractor.image_features[image_fileName_feature]
    greedy_caption = generate_caption(feature_extractor.pred_model, feature_extractor.tokenizer, photo, feature_extractor.max_length)
    
    # Save the greedy caption
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputfiles'))
    os.makedirs(output_dir, exist_ok=True)
    
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data', 'Flicker_Dataset', f'{image_fileName}'))
    img = Image.open(img_path)
    
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    caption_text = ' '.join(greedy_caption)
    wrapped_caption = textwrap.fill(caption_text, width=70)
    plt.title(wrapped_caption, fontsize=12)
    output_file_path = os.path.join(output_dir, f'output_image_with_caption_{model_name}_{model_variant}_greedy.png')
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()
    
    if beam_search:
        # Generate captions for the test image using beam search
        vocab_size = len(feature_extractor.tokenizer.word_index) + 1
        beam_caption, _ = generate_caption_beam(feature_extractor.pred_model, feature_extractor.tokenizer, photo, feature_extractor.max_length, vocab_size, beam_width)
        
        # Save the beam search caption
        plt.imshow(img)
        plt.axis('off')  # Hide axes
        caption_text = ' '.join(beam_caption)
        wrapped_caption = textwrap.fill(caption_text, width=70)
        plt.title(wrapped_caption, fontsize=12)
        output_file_path = os.path.join(output_dir, f'output_image_with_caption_{model_name}_{model_variant}_beam{beam_width}.png')
        plt.savefig(output_file_path, bbox_inches='tight')
        plt.close()
    
    # Calculate BLEU scores
    chencherry = SmoothingFunction()
    ref_cap_reformat = [cap.split()[1:-1] for cap in reference_captions]
    
    bleu_scores = {}
    for i in range(4):
        n = i + 1
        weights = [1.0 / n] * n + [0.0] * (4 - n)
        bleu_scores[f"BLEU-{n}"] = sentence_bleu(ref_cap_reformat, greedy_caption, weights=weights, smoothing_function=chencherry.method1)
    
    if beam_search:
        bleu_scores_beam = {}
        for i in range(4):
            n = i + 1
            weights = [1.0 / n] * n + [0.0] * (4 - n)
            bleu_scores_beam[f"BLEU-{n}"] = sentence_bleu(ref_cap_reformat, beam_caption, weights=weights, smoothing_function=chencherry.method1)
    
    # Print BLEU scores
    print(f"BLEU Scores: {model_name} {model_variant.upper()} Model - Greedy Search")
    for n in range(1, 5):
        print(f"BLEU-{n}: {bleu_scores[f'BLEU-{n}']:.4f}")
    
    if beam_search:
        print(f"\nBLEU Scores: {model_name} {model_variant.upper()} Model - Beam Search (k={beam_width})")
        for n in range(1, 5):
            print(f"BLEU-{n}: {bleu_scores_beam[f'BLEU-{n}']:.4f}")
    
    # Calculate ROUGE scores
    rouge = Rouge()
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)
    
    rouge_scores = scorer.score(' '.join(ref_cap_reformat[0]), ' '.join(greedy_caption))
    
    if beam_search:
        rouge_scores_beam = scorer.score(' '.join(ref_cap_reformat[0]), ' '.join(beam_caption))
    
    # Print ROUGE scores
    print(f"ROUGE Scores (Greedy Search) : {model_name} {model_variant.upper()}")
    for key in rouge_scores:
        print(f'{key.upper()}: {rouge_scores[key].fmeasure:.2f}')
    
    if beam_search:
        print(f"ROUGE Scores (Beam Search) : {model_name} {model_variant.upper()}")
        for key in rouge_scores_beam:
            print(f'{key.upper()}: {rouge_scores_beam[key].fmeasure:.2f}')

if __name__ == "__main__":
    # Run for Inception model with greedy search for both LSTM and BiLSTM
    generate_captions_and_scores(model_name='inception', model_variant='lstm', beam_search=False)
    generate_captions_and_scores(model_name='inception', model_variant='bilstm', beam_search=False)
    
    # Run for VGG model with greedy search for both LSTM and BiLSTM
    generate_captions_and_scores(model_name='vgg', model_variant='lstm', beam_search=False)
    generate_captions_and_scores(model_name='vgg', model_variant='attention', beam_search=False)
    
    # Run for Inception model with beam search for both LSTM and BiLSTM
    generate_captions_and_scores(model_name='inception', model_variant='lstm', beam_search=True, beam_width=5)
    generate_captions_and_scores(model_name='inception', model_variant='bilstm', beam_search=True, beam_width=5)
    
    # Run for VGG model with beam search for both LSTM and BiLSTM
    generate_captions_and_scores(model_name='vgg', model_variant='lstm', beam_search=True, beam_width=5)
