import pandas as pd

from challenge.model import DelayModel

print("Loading data...")
# Read the data
df = pd.read_csv("data/data.csv")
print("-> Data loaded")

# Create the model
model = DelayModel()

print("Preprocessing data...")
# Preprocess the data
X_train, y_train = model.preprocess(df, "delay")
print("-> Preprocessed data")


print("Training model...")
# Train the model
model.fit(X_train, y_train)
print("-> Model trained")

print("Saving model...")
# Store the model
model.save("challenge/tmp/model_checkpoint.pkl")
print("-> Model saved")
