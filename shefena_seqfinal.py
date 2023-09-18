import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow info and warnings

# Load the recipe data from the JSON file
with open('recipes_eng.json', 'r') as file:
    #recipes = json.load(file)
    data = json.load(file)
    all_recipes = data['recipes']
    recipes = [recipe for recipe in all_recipes if
               recipe['recipe'] not in ["Greetings", "Goodbye", "Age", "Name", "Service", "hours", "how", "tradition",
                                        "kulomkililat", "kulom", "kulomketematat", "kulomketema", "kulomtihztotat"]]

# Extract input (recipe name + ingredients) and output (instructions) data
input_data = [f"{recipe['recipe']} Ingredients: {' '.join(recipe['ingredients'])} Start Instructions:" for recipe
              in recipes]
output_data = [' '.join(recipe['instructions']) for recipe in recipes]

# Tokenize the input and output data
tokenizer = Tokenizer(filters='')
tokenizer.fit_on_texts(input_data + output_data)
input_sequences = tokenizer.texts_to_sequences(input_data)
output_sequences = tokenizer.texts_to_sequences(output_data)

# Pad sequences to have the same length
max_seq_length = max(len(seq) for seq in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_seq_length, padding='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_seq_length, padding='post')

# Define the Seq2Seq model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 256, input_length=max_seq_length))
model.add(LSTM(256, return_sequences=True))
model.add(Dense(len(tokenizer.word_index) + 1, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model
model.fit(input_sequences, np.expand_dims(output_sequences, -1), batch_size=32, epochs=2000)

# Test the recipe generation
while True:
    seed_input = input("You have: ")


    # Generate a new recipe
    #seed_input = "Your_Seed_Recipe_Name Ingredients: Your_Seed_Ingredients Start Instructions:"
    # Tokenize the seed input
    seed_tokens = tokenizer.texts_to_sequences([seed_input])[0]
    # Pad the seed input to have the same length as training sequences
    seed_tokens = pad_sequences([seed_tokens], maxlen=max_seq_length, padding='post')

    generated_recipe = []

    for _ in range(max_seq_length):
        # Predict the next token without seeing the timing information
        with tf.device('/cpu:0'):  # Optional: Forces CPU execution to suppress GPU logs
            next_token_probs = model.predict(seed_tokens, verbose=0)

        predicted_token = np.argmax(next_token_probs[0, -1, :])

        # Convert token to word
        predicted_word = tokenizer.index_word.get(predicted_token, '')

        # Add the predicted word to the generated recipe
        generated_recipe.append(predicted_word)

        # Update the seed input
        seed_tokens[0, -1] = predicted_token  # Update the last token in the seed input

    # Print the generated recipe
    print('Generated Recipe:')
    print(' '.join(generated_recipe))
