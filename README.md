# Assignment2
```

```

Initial Model Performance and Issues

At the beginning of the project, the neural network model achieved an accuracy of approximately 69%. Although this result might appear acceptable at first glance, further analysis showed that the model was not learning meaningful patterns from the data. The training and validation accuracy quickly plateaued, which indicated that the model was likely predicting the majority class most of the time.

One major issue was the extremely large loss value during the first training epoch, which suggested numerical instability. This problem was mainly caused by improper data preprocessing, especially the lack of feature scaling and the presence of infinite or very large numeric values.

**Data Preprocessing Improvements**

**1. Handling Invalid and Infinite Values**

Although missing values were initially removed using dropna(), the dataset still contained infinite values and extremely large numbers. These values caused errors when applying feature scaling.

To fix this issue, infinite values were explicitly replaced and removed before scaling:

```
import numpy as np

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()
```

This step ensured that all remaining values were finite and suitable for numerical processing.

**2. Feature Scaling**

In the initial version, feature scaling was not applied, which significantly affected model performance. Neural networks are highly sensitive to the scale of input features, especially when values differ by several orders of magnitude.

Feature normalization was added using StandardScaler:

```
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

After scaling, all features had a mean close to zero and a standard deviation close to one, which stabilized training and improved convergence.

Model Architecture Improvements

The original model used a relatively large architecture without proper preprocessing, which contributed to unstable learning. After improving the data quality, the model architecture was simplified while maintaining sufficient capacity:
```
model = tf.keras.Sequential([
    tf.keras.Input(shape=(X.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

```

This structure was more appropriate for tabular data and reduced unnecessary complexity.

Training Optimization

To prevent unnecessary training once performance stopped improving, early stopping was added:

from tensorflow.keras.callbacks import EarlyStopping
```
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

This helped reduce overtraining and ensured that the best-performing model was retained.

Model Evaluation and Results

After applying the improvements, the model achieved a test accuracy of 99.4%, which represents a significant improvement over the initial results.

To evaluate performance more accurately than accuracy alone, a classification report and confusion matrix were generated:

from sklearn.metrics import classification_report, confusion_matrix

```
y_pred = model.predict(X_test).argmax(axis=1)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```
