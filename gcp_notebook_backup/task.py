import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from tensorflow.keras.layers import Dense, Dropout, Embedding, GRU, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import argparse
import hypertune

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def load_data():
    dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech')
    df = dataset['train'].to_pandas()
    # Use 20% of data for faster training (adjust as needed)
    filtered_df = df[['text', 'insult']].dropna().sample(frac=0.2, random_state=42)

    def map_insult(insult):
        if insult <= 1.0:
            return 0  # Not an insult
        elif 2.0 <= insult <= 3.0:
            return 1  # Insult
        else:
            return 2  # Severe insult

    filtered_df['insult_category'] = filtered_df['insult'].apply(map_insult)
    return filtered_df

def split_data(filtered_df):
    X = filtered_df['text']
    y = filtered_df['insult_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def preprocess_text(X_train, X_test, max_len):
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(X_train)
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=max_len)
    X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=max_len)
    vocab_size = len(tokenizer.word_index) + 1
    return X_train_padded, X_test_padded, vocab_size

def build_model(vocab_size, embedding_dim, max_len, learning_rate, dropout_rate):
    inputs = Input(shape=(max_len,))
    x = Embedding(vocab_size, embedding_dim, input_length=max_len)(inputs)
    x = GRU(32)(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(3, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--max_len', type=int, default=50, help='Maximum sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=7, help='Number of epochs')
    args = parser.parse_args()

    filtered_df = load_data()
    X_train, X_test, y_train, y_test = split_data(filtered_df)
    X_train_padded, X_test_padded, vocab_size = preprocess_text(X_train, X_test, args.max_len)

    model = build_model(vocab_size, args.embedding_dim, args.max_len,
                        args.learning_rate, args.dropout_rate)
    print(model.summary())

    history = model.fit(X_train_padded, y_train,
                        validation_data=(X_test_padded, y_test),
                        batch_size=args.batch_size,
                        epochs=args.epochs)

    test_loss, test_accuracy = model.evaluate(X_test_padded, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Report the tuning metric (e.g., final test accuracy)
    hpt = hypertune.HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='accuracy',
        metric_value=test_accuracy,
        global_step=args.epochs
    )

if __name__ == '__main__':
    main()
