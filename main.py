import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# necessary variables
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000

# Load models
sarcasm_model = tf.keras.models.load_model("Sarcasm_Classifier.h5")
poetry_model = tf.keras.models.load_model("poetrymodel.h5")

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

# Take user input
user_input = str(input("Your Prompt Here:"))

# Convert user input to a list of sentences (not characters)
texts = [user_input]

# Convert the texts into sequences
sequences = tokenizer.texts_to_sequences(texts)
padddd = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Predict sarcasm using the loaded sarcasm model
sarcasm = sarcasm_model.predict(padddd)

next_words = 100

if sarcasm > 0.5:
    print(f"There was sarcasm in the text {sarcasm[0][0]*100:.2f}%")
else:
    # Generate text using the poetry model
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([user_input])[0]
        token_list = pad_sequences([token_list], maxlen=max_length-1, padding='pre')
        
        # Predict the next word index
        predicted = np.argmax(poetry_model.predict(token_list), axis=-1)
        
        # Find the word corresponding to the predicted index
        output_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
                
        user_input += " " + output_word

    print(user_input)