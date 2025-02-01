from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import pandas as pd
import numpy as np

# Load and preprocess the dataset
def load_data():
    from datasets import load_dataset
    dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
    df = dataset['train'].to_pandas()
    
    # Select relevant columns
    filtered_df = df[['text', 'insult']].dropna()

    # Map insult score into categories
    def map_insult(insult):
        if insult <= 1.0:
            return 0  # Not really an insult
        elif 2.0 <= insult <= 3.0:
            return 1  # Insult
        else:
            return 2  # Severe insult

    filtered_df['insult_category'] = filtered_df['insult'].apply(map_insult)
    return filtered_df

# Split dataset into train and test sets
def split_data(filtered_df):
    X = filtered_df['text']
    y = filtered_df['insult_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Tokenize the dataset
def tokenize_data(tokenizer, X_train, X_test, max_len=128):
    train_encodings = tokenizer(
        list(X_train), truncation=True, padding=True, max_length=max_len, return_tensors="tf"
    )
    test_encodings = tokenizer(
        list(X_test), truncation=True, padding=True, max_length=max_len, return_tensors="tf"
    )
    return train_encodings, test_encodings

# Build the model
def build_bert_model():
    model = TFBertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=3  # 3 classes
    )
    # Use TensorFlow-compatible optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    model.compile(optimizer=optimizer, loss=model.compute_loss, metrics=['accuracy'])
    return model

# Main function
def main():
    # Load and preprocess data
    filtered_df = load_data()
    X_train, X_test, y_train, y_test = split_data(filtered_df)

    # Load pre-trained BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    train_encodings, test_encodings = tokenize_data(tokenizer, X_train, X_test)

    # Convert labels to tensors
    y_train_tensor = tf.convert_to_tensor(y_train.tolist())
    y_test_tensor = tf.convert_to_tensor(y_test.tolist())

    # Build BERT model
    model = build_bert_model()
    print(model.summary())

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # Train the model
    history = model.fit(
        [train_encodings['input_ids'], train_encodings['attention_mask']],
        y_train_tensor,
        validation_data=(
            [test_encodings['input_ids'], test_encodings['attention_mask']],
            y_test_tensor,
        ),
        batch_size=16,
        epochs=5,
        callbacks=[early_stopping]
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        [test_encodings['input_ids'], test_encodings['attention_mask']],
        y_test_tensor
    )
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Predict and display results for 5 examples
    predictions = model.predict([test_encodings['input_ids'][:5], test_encodings['attention_mask'][:5]])
    predicted_classes = np.argmax(predictions.logits, axis=1)
    for i in range(5):
        print(f"Text: {X_test.iloc[i]}")
        print(f"Predicted Insult Category: {predicted_classes[i]}")
        print(f"Actual Insult Category: {y_test.iloc[i]}\n")

if __name__ == "__main__":
    main()
