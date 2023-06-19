import pandas as pd
import pickle
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score

# Function to display visually appealing feedback
def display_feedback():
    print("=========================================")
    print("        Code Execution in Progress        ")
    print("=========================================")
    print("")

# Display the visually appealing feedback
display_feedback()

# Add a small delay for effect
time.sleep(2)

# Load the real-life data to analyze
data = pd.read_csv('valid.csv')

# Remove the contents of the label column
data['label'] = ''

# Load the trained vectorizer
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Load the trained model
with open('sentiment_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Apply the trained vectorizer to transform the text data with progress feedback
tqdm.pandas(desc="Transforming text data")
X_real_life = vectorizer.transform(tqdm(data['text'], unit=' samples', dynamic_ncols=True))
predictions = model.predict(X_real_life)

# Output the predictions
data['predicted_sentiment'] = predictions
data.to_csv('predicted_sentiment.csv', index=False)

# Compute and print the accuracy of the predictions
true_labels = data['predicted_sentiment']
accuracy = accuracy_score(true_labels, predictions)
print("Accuracy:", accuracy)

# Display completion message
print("")
print("=========================================")
print("     Code Execution Completed Successfully! ")
print("=========================================")
