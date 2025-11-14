import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
num_words = 10000
maxlen = 200
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
model = Sequential([
 Embedding(input_dim=num_words, output_dim=64, input_length=maxlen),
 LSTM(64, dropout=0.3, recurrent_dropout=0.3),
 Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("\nTraining the model...\n")
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_data=(x_test, y_test), verbose=1)
print("\nModel training complete!\n")
word_index = imdb.get_word_index()
def encode_review(text):
 words = text.lower().split()
 encoded = [1]
 for w in words:
 encoded.append(word_index.get(w, 2))
 return pad_sequences([encoded], maxlen=maxlen)
while True:
 review = input("Enter a movie review (or type 'exit' to quit): ")
 if review.lower() == 'exit':
 print("Exiting the program. Goodbye!")
 break
 pred = model.predict(encode_review(review))
print("Predicted Sentiment:", "Positive 😊" if pred[0][0] > 0.5 else "Negative 😞")
  
