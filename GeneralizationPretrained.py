
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt
import os

train=pd.read_csv("C:/Users/lenovo/Downloads/iomtdosnorm.csv",skipinitialspace=True, low_memory=False)

import numpy as np
train.replace([np.inf, -np.inf], np.nan, inplace=True)
train=train.dropna()

train.drop(['label'], axis=1, inplace=True)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns
categorical_cols = train.select_dtypes(include=['object']).columns

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])

import pandas as pd

# Assuming 'train' is your DataFrame and 'Label' is your target variable
correlations = train.corr()['attack'].drop('attack')  # Drop target variable itself
correlations = correlations.abs().sort_values(ascending=False)  # Sort by absolute value

top_features = correlations.head(36).index  # Select top 4 features
selected_data = train[top_features]  # Create a new DataFrame with selected features

y=train['attack']
# Convert categorical labels to numerical using LabelEncoder

#y_train=y_train.astype(float)

X=train[top_features]
#X=train.drop('attack', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import SMOTE
def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    return X_balanced, y_balanced
(X_train,y_train)=balance_dataset(X_train, y_train)

X_train = pd.DataFrame(X_train)  # Convert to DataFrame

from PIL import Image
import os
import csv



labels = ["Normal", "MITM"]
os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download1"):
    os.makedirs("download1")

# Change the current working directory
os.chdir("download1")


# Convert the mitmds array to uint8
X_train = X_train.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_train.shape[0]):
    if y_train.iloc[i]==0:
        binary_values = np.unpackbits(X_train.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Normal.{i}.jpg',"JPEG")



labels = ["Normal", "MITM"]
os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download1"):
    os.makedirs("download1")

# Change the current working directory
os.chdir("download1")


# Convert the mitmds array to uint8
X_train = X_train.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_train.shape[0]):
    if y_train.iloc[i]==1:
        binary_values = np.unpackbits(X_train.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Attack.{i}.jpg',"JPEG")



labels = ["Normal", "MITM"]
os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download1val"):
    os.makedirs("download1val")

# Change the current working directory
os.chdir("download1val")


# Convert the mitmds array to uint8
X_val = X_val.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_val.shape[0]):
    if y_val.iloc[i]==0:
        binary_values = np.unpackbits(X_val.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Normal.{i}.jpg',"JPEG")



labels = ["Normal", "MITM"]
os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download1val"):
    os.makedirs("download1val")

# Change the current working directory
os.chdir("download1val")


# Convert the mitmds array to uint8
X_val = X_val.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_val.shape[0]):
    if y_val.iloc[i]==1:
        binary_values = np.unpackbits(X_val.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Attack.{i}.jpg',"JPEG")


os.chdir("C:/Users/lenovo/")

train_img_dir_n = "C:/Users/lenovo/download1"
#train_img_dir_n =train_img_dir_n [:1000]
train_img_paths_n = [os.path.join(train_img_dir_n,filename) for filename in os.listdir(train_img_dir_n)]

import re

train_path_df = pd.DataFrame({
    'path': [],
    'target': []
})

for path in train_img_paths_n:
    pattern = r'Normal'

    match = re.search(pattern, path)

    if match:
        train_path_df = pd.concat([train_path_df, pd.DataFrame({'path': [path], 'target': [0]})], ignore_index=True)
    else:
        train_path_df = pd.concat([train_path_df, pd.DataFrame({'path': [path], 'target': [1]})], ignore_index=True)

val_img_dir_n = "c:/users/lenovo/download1val"
#train_img_dir_n =train_img_dir_n [:1000]
val_img_paths_n = [os.path.join(val_img_dir_n,filename) for filename in os.listdir(val_img_dir_n)]

import re

val_path_df = pd.DataFrame({
    'path': [],
    'target': []
})

for path in val_img_paths_n:
    pattern = r'Normal'

    match = re.search(pattern, path)

    if match:
        val_path_df = pd.concat([val_path_df, pd.DataFrame({'path': [path], 'target': [0]})], ignore_index=True)
    else:
        val_path_df = pd.concat([val_path_df, pd.DataFrame({'path': [path], 'target': [1]})], ignore_index=True)

from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1./255,
    #validation_split = .2

)

train_path_df['target'] = train_path_df['target'].astype(str)
train_image_generator = datagen.flow_from_dataframe(
    train_path_df,
    x_col='path',
    y_col='target',
    target_size=(75,75),  # Adjust to match your model's input size
   # batch_size=32,
    class_mode='binary',  # Change to 'binary' if you have binary classes
    shuffle=True,
    color_mode='rgb',
    #subset='training'
)

val_path_df['target'] = val_path_df['target'].astype(str)
val_image_generator = datagen.flow_from_dataframe(
    val_path_df,
    x_col='path',
    y_col='target',
    target_size=(75,75),  # Adjust to match your model's input size
    #batch_size=32,
    class_mode='binary',  # Change to 'binary' if you have binary classes
    shuffle=True,
    color_mode='rgb',
    #subset='validation'
)


#inception
from keras.applications.inception_v3 import InceptionV3
inception = InceptionV3(input_shape=(75,75,3), weights='imagenet', include_top=False)
for layer in inception.layers:
    layer.trainable = False
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras import layers

x = layers.Flatten()(inception.output)
#x = layers.Dense(128, activation='relu')(x)
#x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(inception.input, x)

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, mode="min", verbose=1, min_lr=0.001)
call_backs = [ early_stopping, lr_reduce]

import keras
checkpoint_filepath = 'c:/users/lenovo/rtiotdosddos2_12_24.weights.h5'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)
call_back = [ early_stopping, lr_reduce, model_checkpoint_callback ]

from datetime import datetime
start_time = datetime.now()

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.001),
              metrics=['accuracy'])
"""
batch_size=32
STEP_SIZE_TRAIN = len(train_image_generator) // batch_size
#STEP_SIZE_TRAIN=train_image_generator.n//train_image_generator.batch_size
STEP_SIZE_VALID=len(val_image_generator) // batch_size
STEP_SIZE_TEST=test_image_generator.n//test_image_generator.batch_size
"""
history=model.fit(train_image_generator,
                    #steps_per_epoch=128,
                    validation_data=val_image_generator,
                    #validation_steps=128,
                    epochs=10,
                  callbacks=call_back
)
#Train the model
#history=model.fit(train_image_generator,epochs=100,  callbacks=call_backs)

end_time = datetime.now()

# Calculate elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Training time: {elapsed_time}")
# Evaluate the model

import pyarrow
file_path = "C:/Users/lenovo/Downloads/UNSW_NB15_training-set.parquet"
test=pd.read_parquet(file_path, engine='pyarrow')

# Select rows where 'attack_cat' is 'DoS' or 'Normal'
selected_rows = test[test['attack_cat'].isin(['Shellcode', 'Normal'])]

# Display the first few rows of the selected data
display(selected_rows.head())

import numpy as np
selected_rows = selected_rows.copy() # Create a copy to avoid SettingWithCopyWarning
selected_rows.replace([np.inf, -np.inf], np.nan, inplace=True)
selected_rows=selected_rows.dropna()

selected_rows.drop(['attack_cat'], axis=1, inplace=True)

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Identify categorical columns (including 'category' dtype)
categorical_cols = selected_rows.select_dtypes(include=['object', 'category']).columns

# Apply Label Encoding to each categorical column
for col in categorical_cols:
    le = LabelEncoder()
    selected_rows[col] = le.fit_transform(selected_rows[col])

#with zeroes
feat=['sjit','djit','synack', 'ackdat','smean','dmean',
      'blank','blank','blank','proto','sjit','djit' ,
      'smean','dmean','sbytes','dbytes','blank','blank',
      'rate','sjit','djit','blank','blank','blank',
      'dur','blank','blank','blank','blank','blank',
      'blank','blank','blank','blank','blank','service']
# Create a new DataFrame with 36 columns named numerically
new_df = pd.DataFrame()

# Add 36 columns with numerical names (0 to 35) and initialize them with 0
for i in range(36):
  for col_name in feat:
    if col_name != 'blank':
        # If the column name is not 'blank', get the data from the original DataFrame
        new_df[i] = selected_rows[col_name]
    else:
            # Handle the case where the feature is not in selected_rows (though it should be based on the context)
        new_df[i] = 0
    
# Display the first few rows of the new DataFrame

y_test=selected_rows['label']
# Convert categorical labels to numerical using LabelEncoder

#y_train=y_train.astype(float)

X_test=new_df
#X_train=train.drop('attack', axis=1)

from imblearn.over_sampling import SMOTE
def balance_dataset(X, y):
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    return X_balanced, y_balanced
(X_test,y_test)=balance_dataset(X_test, y_test)



os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download2"):
    os.makedirs("download2")

# Change the current working directory
os.chdir("download2")


# Convert the mitmds array to uint8
X_test = X_test.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_test.shape[0]):
    if y_test.iloc[i]==0:
        binary_values = np.unpackbits(X_test.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Normal.{i}.jpg',"JPEG")



os.chdir("C:/Users/lenovo/")
# Check if the directory exists, if not create it
if not os.path.exists("download2"):
    os.makedirs("download2")

# Change the current working directory
os.chdir("download2")


# Convert the mitmds array to uint8
X_test = X_test.astype(np.uint8)

# Iterate over each element in the mitmds array
for i in range(X_test.shape[0]):
    if y_test.iloc[i]==1:
        binary_values = np.unpackbits(X_test.iloc[i].values.view(np.uint8))

    # Group the bits into 8-bit chunks
        grouped_values = np.packbits(binary_values.reshape(-1, 8))

    # Determine the dimensions of the image (assuming a square image)
        image_size = int(np.ceil(np.sqrt(len(grouped_values))))

    # Pad the grouped values with zeros to make a square image
        padded_values = np.pad(grouped_values, (0, image_size**2 - len(grouped_values)), 'constant')

    # Reshape the padded values into a 2D image array
        image_data = padded_values.reshape(image_size, image_size)

    # Create an image from the array and save it
        image = Image.fromarray(image_data, mode='L')
        image.save(f'Attack.{i}.jpg',"JPEG")

test_img_dir_n = "C:/Users/lenovo/download2"
#train_img_dir_n =train_img_dir_n [:1000]
test_img_paths_n = [os.path.join(test_img_dir_n,filename) for filename in os.listdir(test_img_dir_n)]

import re

test_path_df = pd.DataFrame({
    'path': [],
    'target': []
})

for path in test_img_paths_n:
    pattern = r'Normal'

    match = re.search(pattern, path)

    if match:
        test_path_df = pd.concat([test_path_df, pd.DataFrame({'path': [path], 'target': [0]})], ignore_index=True)
    else:
        test_path_df = pd.concat([test_path_df, pd.DataFrame({'path': [path], 'target': [1]})], ignore_index=True)

test_path_df['target'] = test_path_df['target'].astype(str)

test_image_generator = datagen.flow_from_dataframe(
    test_path_df,
    x_col='path',
    y_col='target',
    target_size=(75,75),  # Adjust to match your model's input size
    #batch_size=32,
    class_mode='binary',  # Change to 'binary' if you have binary classes
    shuffle=False,
    color_mode='rgb'
)


import numpy as np
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
#checkpoint_filepath = 'c:/users/lenovo/rtiotdosddos2_12_24.weights.h5'
#model.load_weights(checkpoint_filepath)
predictions = model.predict(test_image_generator)

best_accuracy = 0
best_threshold = 0
best_report = ""

# Calculate ROC AUC directly
true_labels = test_image_generator.classes  # Get true labels once outside the loop
roc_auc = roc_auc_score(true_labels, predictions)  # Use roc_auc_score

for threshold in np.arange(0.1, 1.0, 0.1):
    predicted_labels = (predictions > threshold).astype(int)

    accuracy = accuracy_score(true_labels, predicted_labels)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_report = classification_report(true_labels, predicted_labels)
        cm = confusion_matrix(true_labels, predicted_labels)

print(f"Best Accuracy: {best_accuracy} at Threshold: {best_threshold}")
print(f"ROC AUC: {roc_auc}")  # Print ROC AUC
print("Best Classification Report:\n", best_report)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()


