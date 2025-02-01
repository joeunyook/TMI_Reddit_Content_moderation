from datasets import load_dataset

# Load the dataset
dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech', 'default')
df = dataset['train'].to_pandas()

# Filter the columns related to insult
columns_of_interest = ['text', 'insult', 'sentiment', 'humiliate', 'status']
filtered_df = df[columns_of_interest]

# Display the first 5 rows
print(filtered_df.head())
