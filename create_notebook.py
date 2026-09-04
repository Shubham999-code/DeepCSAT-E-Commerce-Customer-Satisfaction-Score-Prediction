import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown and code cells
cells = []

# Section 1: Introduction
cells.append(nbf.v4.new_markdown_cell("""# Customer Satisfaction (CSAT) Prediction using Deep Learning
## e-Commerce Customer Support Data

This notebook builds an Artificial Neural Network (ANN) to predict the Customer Satisfaction (CSAT) score (1-5) based on customer interaction features. 
This notebook is designed for **Google Colab**.

### Evaluation Criteria Covered:
1. Data Integrity and Cleaning
2. Feature Engineering and Selection
3. Data Preprocessing and Transformation
4. Model Development and Architecture
5. Training Efficiency and Optimization
6. Evaluation Metrics and Model Validation
"""))

# Section 2: Imports
cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# If using Google Colab, uncomment the line below to mount drive if dataset is on Drive
# from google.colab import drive
# drive.mount('/content/drive')
"""))

# Section 3: Data Loading
cells.append(nbf.v4.new_markdown_cell("""## 1. Data Loading & Understanding"""))
cells.append(nbf.v4.new_code_cell("""# Load the dataset
# Adjust the path based on where you upload the file in Colab
# E.g., df = pd.read_csv('/content/eCommerce_Customer_support_data.csv')
file_path = 'eCommerce_Customer_support_data.csv'
try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
    print("Shape:", df.shape)
except FileNotFoundError:
    print(f"File {file_path} not found. Please upload the dataset to Colab.")

# Display basic info
if 'df' in locals():
    display(df.head())
    display(df.info())
"""))

# Section 4: Data Cleaning & Integrity
cells.append(nbf.v4.new_markdown_cell("""## 2. Data Integrity and Cleaning"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # 1. Drop irrelevant columns
    # Unique id and Order_id are unique identifiers, not useful for prediction
    # Agent names and Manager names might cause overfitting due to high cardinality, let's keep 'Agent Shift' and 'Tenure Bucket' instead.
    cols_to_drop = ['Unique id', 'Order_id', 'Agent_name', 'Supervisor', 'Manager', 'Survey_response_Date', 'Customer_City', 'Product_category']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # 2. Check for missing values
    print("Missing values before cleaning:")
    print(df.isnull().sum())

    # 3. Handle Missing Values
    # 'Customer Remarks' is mostly null unless the customer left a review. We'll handle this in feature engineering.
    # 'Item_price' and 'connected_handling_time' might have nulls. We can fill with median.
    if 'Item_price' in df.columns:
        df['Item_price'] = df['Item_price'].fillna(df['Item_price'].median())
    
    if 'connected_handling_time' in df.columns:
        df['connected_handling_time'] = df['connected_handling_time'].fillna(df['connected_handling_time'].median())
        
    # Drop rows where target CSAT Score is null
    df = df.dropna(subset=['CSAT Score'])
    
    print("\\nMissing values after initial cleaning:")
    print(df.isnull().sum())
"""))

# Section 5: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 3. Feature Engineering and Selection"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # 1. Text Feature: Has Remarks
    # Let's create a binary feature indicating if the customer left a remark
    df['has_remarks'] = df['Customer Remarks'].notnull().astype(int)
    
    # We can now drop 'Customer Remarks' to keep the model strictly tabular
    df = df.drop(columns=['Customer Remarks'])

    # 2. Time Features: Resolution Time
    # Convert dates to datetime
    if 'Issue_reported at' in df.columns and 'issue_responded' in df.columns:
        df['Issue_reported at'] = pd.to_datetime(df['Issue_reported at'], errors='coerce')
        df['issue_responded'] = pd.to_datetime(df['issue_responded'], errors='coerce')
        
        # Calculate time taken to respond in minutes
        df['response_time_mins'] = (df['issue_responded'] - df['Issue_reported at']).dt.total_seconds() / 60.0
        
        # Fill missing response times with the median
        df['response_time_mins'] = df['response_time_mins'].fillna(df['response_time_mins'].median())
        
        # Drop the original timestamp columns
        df = df.drop(columns=['Issue_reported at', 'issue_responded', 'order_date_time'], errors='ignore')

    # Ensure CSAT Score is an integer
    df['CSAT Score'] = df['CSAT Score'].astype(int)
    
    print("Features engineered successfully!")
    display(df.head())
"""))

# Section 6: Data Preprocessing
cells.append(nbf.v4.new_markdown_cell("""## 4. Data Preprocessing and Transformation"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # Define features (X) and target (y)
    X = df.drop(columns=['CSAT Score'])
    # CSAT Score is 1 to 5. We shift it to 0 to 4 for Keras categorical encoding.
    y = df['CSAT Score'] - 1 

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64', 'int32']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Fill any remaining NaNs in categorical with 'Unknown'
    for col in categorical_cols:
        X[col] = X[col].fillna('Unknown')

    print("Numerical Columns:", numerical_cols)
    print("Categorical Columns:", categorical_cols)

    # Preprocessing Pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Split the data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.15, random_state=42, stratify=y_train_val)

    # Fit and transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    # Convert y to categorical (one-hot encoding for the target)
    num_classes = 5
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"X_train shape: {X_train_processed.shape}")
    print(f"y_train_cat shape: {y_train_cat.shape}")
    
    # Save the preprocessor for local deployment
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("Preprocessor saved as 'preprocessor.pkl'")
"""))

# Section 7: Model Architecture
cells.append(nbf.v4.new_markdown_cell("""## 5. Model Development and Architecture"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    input_dim = X_train_processed.shape[1]

    # Build the ANN model
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3), # Regularization to prevent overfitting
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax') # Softmax for multi-class classification
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()
"""))

# Section 8: Training
cells.append(nbf.v4.new_markdown_cell("""## 6. Training Efficiency and Optimization"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # Callbacks for optimization
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=3, 
        min_lr=1e-6,
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train_processed, y_train_cat,
        validation_data=(X_val_processed, y_val_cat),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
"""))

# Section 9: Evaluation
cells.append(nbf.v4.new_markdown_cell("""## 7. Evaluation Metrics and Model Validation"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()
    plt.show()

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test_processed, y_test_cat, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predictions
    y_pred_prob = model.predict(X_test_processed)
    y_pred = np.argmax(y_pred_prob, axis=1)

    print("\\nClassification Report:")
    # Mapping back to CSAT 1-5
    target_names = ['CSAT 1', 'CSAT 2', 'CSAT 3', 'CSAT 4', 'CSAT 5']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
"""))

# Section 10: Export Model
cells.append(nbf.v4.new_markdown_cell("""## 8. Export for Local Deployment"""))
cells.append(nbf.v4.new_code_cell("""if 'df' in locals():
    # Save the ANN model
    model.save('csat_ann_model.h5')
    print("Model saved as 'csat_ann_model.h5'")
    
    # Preprocessor is already saved as 'preprocessor.pkl'
    print("Download 'csat_ann_model.h5' and 'preprocessor.pkl' for use in the Streamlit app.")
"""))

nb['cells'] = cells

with open('CSAT_Prediction.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
