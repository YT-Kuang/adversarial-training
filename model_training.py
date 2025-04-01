import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from art.attacks.evasion import FastGradientMethod, BasicIterativeMethod, ProjectedGradientDescent
from art.estimators.classification import TensorFlowV2Classifier
from art.utils import to_categorical

# Load dataset
df = pd.read_csv('dataset/UNSW_NB15_training-set.csv')

# Drop rows with null value and duplicates
df = df.dropna().drop_duplicates()
df = df.drop(columns=['attack_cat','id'])

# Convert non-numeric columns to numeric
categorical_columns = ['proto', 'service', 'state']
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Separate features and target
X = df.drop(columns=['label'])
y = df['label']

# Normalize features
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Feature importance using Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)
feature_importances = model.feature_importances_

# Select top 8 features
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

top_features = importance_df['Feature'].head(8).tolist()
X_scaled = X_scaled[top_features]

# Reshape data for 2D CNN
X_scaled_reshaped = X_scaled.to_numpy().reshape(-1, 8, 1, 1)

# Split dataset
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled_reshaped, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Convert labels to categorical
y_train_cat = to_categorical(y_train)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)

# Define 2D CNN model
def build_cnn_model(input_shape):
    model = Sequential([
        # First level
        Conv2D(32, (3, 1), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 1)),
        
        # Second level
        Conv2D(64, (3, 1), activation='relu'),
        Flatten(),
        
        # Full-connected layer
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create CNN model
cnn_model = build_cnn_model(input_shape=(8, 1, 1))

# Train CNN model
cnn_model.fit(X_train, y_train_cat, epochs=20, batch_size=32, validation_data=(X_val, y_val_cat))

# Evaluate baseline CNN
baseline_accuracy = cnn_model.evaluate(X_test, y_test_cat, verbose=0)[1]
print(f"Baseline CNN Accuracy: {baseline_accuracy:.2f}")

# Wrap CNN with ART
classifier = TensorFlowV2Classifier(
    model=cnn_model,
    nb_classes=2,
    input_shape=(8, 1, 1),
    loss_object=tf.keras.losses.CategoricalCrossentropy()
)

# Generate adversarial samples
def generate_adversarial_samples(classifier, X, method):
    if method == "FGSM":
        attack = FastGradientMethod(estimator=classifier, eps=0.2)
    elif method == "BIM":
        attack = BasicIterativeMethod(estimator=classifier, eps=0.2, max_iter=10)
    elif method == "PGD":
        attack = ProjectedGradientDescent(estimator=classifier, eps=0.2, max_iter=10)
    else:
        raise ValueError("Unknown attack method")
    return attack.generate(X)

# Test CNN on adversarial samples
adversarial_methods = ["FGSM", "BIM", "PGD"]
for method in adversarial_methods:
    X_test_adv = generate_adversarial_samples(classifier, X_test, method)
    y_pred_adv = np.argmax(classifier.predict(X_test_adv), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    adv_accuracy = accuracy_score(y_true, y_pred_adv)
    print(f"Accuracy on {method} adversarial samples: {adv_accuracy:.2f}")

# Adversarial training with min-max formulation
for method in adversarial_methods:
    X_train_adv = generate_adversarial_samples(classifier, X_train, method)
    X_combined = np.vstack([X_train, X_train_adv])
    y_combined = np.vstack([y_train_cat, y_train_cat])
    cnn_model.fit(X_combined, y_combined, epochs=10, batch_size=32, validation_data=(X_val, y_val_cat))

# Test robust CNN on adversarial samples
for method in adversarial_methods:
    X_test_adv = generate_adversarial_samples(classifier, X_test, method)
    y_pred_adv = np.argmax(classifier.predict(X_test_adv), axis=1)
    y_true = np.argmax(y_test_cat, axis=1)
    adv_accuracy = accuracy_score(y_true, y_pred_adv)
    print(f"Robust CNN Accuracy on {method} adversarial samples: {adv_accuracy:.2f}")