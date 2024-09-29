import pandas as pd
import numpy as np

# read emoticon dataset
train_emoticon_df = pd.read_csv("/home/belief/Desktop/MLAss1/mini-project-1/datasets/train/train_emoticon.csv")
train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
train_emoticon_Y = train_emoticon_df['label'].tolist()
test_emoticon_X = pd.read_csv("/home/belief/Desktop/MLAss1/mini-project-1/datasets/test/test_emoticon.csv")['input_emoticon'].tolist()

print(f"Train dataset size: ")
print(f"train_emoticon_X: {len(train_emoticon_X)} train_emoticon_Y: {len(train_emoticon_Y)}")

print()
print("Test dataset size: ")
print(f"test_emoticon_X: {len(test_emoticon_X)}")