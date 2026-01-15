import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("processed.csv")

X = df.drop('Label1', axis=1)
y = df['Label1']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train
model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=512,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Detailed evaluation
y_pred = model.predict(X_test).argmax(axis=1)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))



#Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))