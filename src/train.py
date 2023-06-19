import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm, tqdm_notebook
import time
import pickle

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

# Load the dataset
data = pd.read_csv('Train.csv')

# Preprocess the data
X = data['text']
y = data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using TF-IDF with progress feedback
vectorizer = TfidfVectorizer()

# Set the progress bar style
tqdm.pandas(desc="Vectorizing training data", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

X_train = vectorizer.fit_transform(tqdm(X_train, unit=' samples', dynamic_ncols=True))
X_test = vectorizer.transform(tqdm(X_test, unit=' samples', dynamic_ncols=True))

# Save the trained vectorizer
with open('vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorizer, file)

# Train a logistic regression model with progress feedback
model = LogisticRegression()

# Set the progress bar style
tqdm.pandas(desc="Training model", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}")

model.fit(X_train, y_train)

# Save the trained model
with open('sentiment_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Display completion message
print("")
print("=========================================")
print("     Code Execution Completed Succesfully! ")
print("=========================================")
