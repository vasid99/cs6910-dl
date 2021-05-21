# Assignment 03 - Recurrent Neural Networks

# Dataset Loading and Preprocessing
- A start and end character is added to the Devanagri words.
- Vocabulary is made for the Roman and Devanagari characters

# Functions
## Dataset Loading and processing
- `str_to_numarray`(strings, char_to_int_map): 
  Converts the atrings to integer arrays
- `vectorize_dataset`(data):
  
  Prepares data for training a seq2seq model.
  - Converts the strings of Roman words to an array of integers to be used as encoder input.
  - Converts the words in Devanagri to an array of integers to be used for decoder input and (one-hot) decoder target (=`decoder_input[1:]`).

## Model for training
- `fresh_seq2seq_model`(network_parameters):
  - Creates the encoder with the embedding layer and RNNs
  - Creates the decoder with the embedding layer, RNNs (, attention layer if `dec_attention=True`) and the dense layer. The initial state is the encoder final state.
  - Returns the model = `tf.keras.Model([encoder_input, decoder_input], decoder_output)`

## Inference
- `extract_inference_submodels`(model):
  
  Rebuilds the encoder and decoder models from the complete model and returns them.

- `seq2seq`(input_words, models, beam_size=1, **kwargs):

  - Performs the seq2seq operation using trained models
  - Inputs:
    - input_words: input words to be transformed
    - models: encoder and decoder models to be used
    - beam_size: size to be used to perform beam search. Default is 1 (greedy decoding)
  - Outputs:
    - output_seq: transformed sequences
    - attn_weights: attention weights; pass `return_attention_weights=True` to return
 
- `prediction_accuracy`(output_data, target_data):
  
  Returns number of exact matches between predictions and target dataset

# Instructions
For training the model, run the subsection "Sample Runs/Training The Model". For training the model using a parameter sweep, run the section "Training Sweep".

To perform evaluation of the model, run the subsection "Sample Runs/Testing the Trained Model". For connectivity and heatmaps, cells within their respective sections can be run.
