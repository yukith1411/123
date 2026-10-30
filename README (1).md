# Deep Learning Final Laboratory — Solutions

Complete Python solutions for the 10 experiments in **DL End semester qp 2026-1**.

## Setup and usage

Run each experiment independently in Google Colab or Jupyter Notebook. A GPU is recommended for neural network experiments. Initial dataset and pretrained weight downloads require internet.

In Colab, install dependencies in a separate cell:

```python
!pip -q install tensorflow scikit-learn matplotlib pillow torch torchvision
```

For local Python, run the same command in a terminal without `!`.

> Record the actual metrics and screenshots from your execution. Results depend on training and environment. These examples have been syntax-checked; full model training was not executed while preparing this README.

## Contents

1. [Compare classification algorithms](#1-compare-classification-algorithms)
2. [Feed-forward neural network](#2-feed-forward-neural-network)
3. [NumPy forward propagation](#3-numpy-forward-propagation)
4. [CNN image classification](#4-cnn-image-classification)
5. [Conv1D text classification](#5-conv1d-text-classification)
6. [Traffic object detection](#6-traffic-object-detection)
7. [SimpleRNN forecasting](#7-simplernn-forecasting)
8. [Restricted Boltzmann Machine](#8-restricted-boltzmann-machine)
9. [LSTM sentiment classification](#9-lstm-sentiment-classification)
10. [Variational Autoencoder](#10-variational-autoencoder)

## 1. Compare classification algorithms

**Aim:** Compare KNN and Random Forest on the Breast Cancer Wisconsin dataset.

**Algorithm:** Load data → stratified split → scale KNN inputs → train classifiers → predict → compare accuracy, macro precision, recall and F1 → display confusion matrices.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, ConfusionMatrixDisplay)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2,
    random_state=42, stratify=data.target
)
models = {
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(5)),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}
results = {}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    scores = {
        "Accuracy": accuracy_score(y_test, predicted),
        "Precision": precision_score(y_test, predicted, average="macro"),
        "Recall": recall_score(y_test, predicted, average="macro"),
        "F1": f1_score(y_test, predicted, average="macro")
    }
    results[name] = scores["F1"]
    print("\n" + name)
    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")
    ConfusionMatrixDisplay.from_predictions(
        y_test, predicted, display_labels=data.target_names,
        cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.show()
best = max(results.values())
print("Best by test macro F1:", ", ".join(
    name for name, score in results.items() if np.isclose(score, best)
))
```

**Output:** Four evaluation metrics, two confusion matrices and the better model by macro F1, including ties.

**Result:** Two supervised classifiers are implemented and compared on the same test set.

## 2. Feed-forward neural network

**Aim:** Classify MNIST handwritten digits using dense layers.

**Algorithm:** Normalize → flatten → dense hidden layers → softmax output → train → evaluate → display predictions.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
model = Sequential([
    layers.Input(shape=(28, 28)), layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}; accuracy: {accuracy:.4f}")
predictions = model.predict(x_test[:10], verbose=0).argmax(axis=1)
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Actual: {y_test[i]}\nPredicted: {predictions[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()
```

**Output:** Test loss, accuracy and sample predictions.

**Result:** A feed-forward network is trained for digit recognition.

## 3. NumPy forward propagation

**Aim:** Demonstrate XOR forward propagation using ReLU and sigmoid.

**Algorithm:** Define inputs and fixed weights → weighted hidden sum → ReLU → weighted output sum → sigmoid → threshold.

The weights are manually chosen to represent XOR. No backpropagation or training is performed.

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([0, 1, 1, 0])
W1 = np.array([[1, -1], [-1, 1]], dtype=float)
b1 = np.zeros(2)
W2 = np.array([[10], [10]], dtype=float)
b2 = np.array([-5], dtype=float)
Z1 = X @ W1 + b1
A1 = relu(Z1)
Z2 = A1 @ W2 + b2
probabilities = sigmoid(Z2).ravel()
predicted = (probabilities >= 0.5).astype(int)
print("Hidden weighted sums:\n", Z1)
print("Hidden activations:\n", A1)
print("Input | Actual | Probability | Predicted")
for inputs, actual, probability, prediction in zip(X.astype(int), y, probabilities, predicted):
    print(f"{inputs} | {actual} | {probability:.4f} | {prediction}")
```

**Expected predictions:** `0, 1, 1, 0`.

**Expected probabilities:** Approximately `0.0067, 0.9933, 0.9933, 0.0067`.

**Result:** Forward propagation produces the XOR truth table.

## 4. CNN image classification

**Aim:** Classify Fashion-MNIST images using convolution and pooling layers.

**Algorithm:** Normalize and add channel dimension → convolution → pooling → convolution → pooling → flatten → dense classification → evaluate.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential, layers

tf.keras.utils.set_random_seed(42)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32")[..., np.newaxis] / 255.0
x_test = x_test.astype("float32")[..., np.newaxis] / 255.0
names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
         "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
model = Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation="relu"), layers.MaxPooling2D(2),
    layers.Flatten(), layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}; accuracy: {accuracy:.4f}")
predicted = model.predict(x_test[:9], verbose=0).argmax(axis=1)
plt.figure(figsize=(10, 9))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i, :, :, 0], cmap="gray")
    plt.title(f"Actual: {names[y_test[i]]}\nPredicted: {names[predicted[i]]}", fontsize=10)
    plt.axis("off")
plt.tight_layout()
plt.show()
```

**Output:** Test metrics and actual versus predicted clothing labels.

**Result:** A CNN is trained to classify clothing images.

## 5. Conv1D text classification

**Aim:** Predict IMDb review sentiment using a one-dimensional CNN.

**Algorithm:** Decode IMDb reviews → clean, tokenize and pad → embedding → Conv1D → global max pooling → binary classification → evaluate new review.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)
VOCAB_SIZE, MAX_LENGTH = 10000, 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
word_index = tf.keras.datasets.imdb.get_word_index()
index_word = {index + 3: word for word, index in word_index.items()}
def decode(sequence):
    return " ".join(index_word.get(int(i), "unknown") for i in sequence if i != 1)
train_text = np.array([decode(s) for s in x_train])
test_text = np.array([decode(s) for s in x_test])
indices = np.random.default_rng(42).permutation(len(train_text))
val_indices, fit_indices = indices[:2500], indices[2500:]
vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE, standardize="lower_and_strip_punctuation",
    output_mode="int", output_sequence_length=MAX_LENGTH
)
vectorizer.adapt(train_text[fit_indices])
inputs = tf.keras.Input(shape=(), dtype=tf.string)
x = vectorizer(inputs)
x = layers.Embedding(VOCAB_SIZE, 64)(x)
x = layers.Conv1D(64, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(32, activation="relu")(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
def dataset(text, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((text, labels))
    if training:
        ds = ds.shuffle(len(text), seed=42)
    return ds.batch(128).prefetch(tf.data.AUTOTUNE)
model.fit(
    dataset(train_text[fit_indices], y_train[fit_indices], True),
    epochs=4,
    validation_data=dataset(train_text[val_indices], y_train[val_indices])
)
loss, accuracy = model.evaluate(dataset(test_text, y_test), verbose=0)
print(f"Test loss: {loss:.4f}; accuracy: {accuracy:.4f}")
review = "The movie was excellent with wonderful acting."
p = float(model.predict(tf.constant([review]), verbose=0)[0, 0])
print("Review:", review)
print("Positive probability:", round(p, 4))
print("Sentiment:", "Positive" if p >= 0.5 else "Negative")
```

**Output:** Test accuracy and new-review sentiment.

**Result:** A Conv1D network is trained for binary text classification.

## 6. Traffic object detection

**Aim:** Detect traffic objects using pretrained Faster R-CNN.

**Algorithm:** Load traffic image → transform → run detector → filter road classes and confidence → draw boxes and labels.

Upload a traffic image named **traffic.jpg** into the current working directory. The pretrained model uses COCO annotations; no new annotation or training is required for this inference demonstration.

```python
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn(weights=weights).to(device).eval()
image = Image.open("traffic.jpg").convert("RGB")
tensor = weights.transforms()(image).to(device)
with torch.inference_mode():
    prediction = model([tensor])[0]
categories = weights.meta["categories"]
road_classes = {"person", "bicycle", "car", "motorcycle", "bus", "truck",
                "traffic light", "stop sign"}
annotated = image.copy()
draw = ImageDraw.Draw(annotated)
count = 0
for box, label, score in zip(prediction["boxes"].cpu(),
                             prediction["labels"].cpu(),
                             prediction["scores"].cpu()):
    name, confidence = categories[int(label)], float(score)
    if confidence < 0.6 or name not in road_classes:
        continue
    x1, y1, x2, y2 = box.tolist()
    draw.rectangle((x1, y1, x2, y2), outline="red", width=3)
    draw.text((x1, max(0, y1 - 14)), f"{name}: {confidence:.2f}", fill="red")
    print(f"{name}: {confidence:.3f}")
    count += 1
print("Detected road objects:", count)
plt.figure(figsize=(14, 9))
plt.imshow(annotated)
plt.title("Traffic Object Detection")
plt.axis("off")
plt.show()
annotated.save("traffic_detected.jpg")
```

**Output:** Annotated traffic image, bounding boxes, labels and confidence scores. Detection depends on the uploaded scene.

**Result:** Actual object detection is demonstrated using a pretrained CNN detector.

## 7. SimpleRNN forecasting

**Aim:** Predict the next sine-wave value using SimpleRNN.

**Algorithm:** Generate series → chronological split → fit scaler on training period → create input windows → train → evaluate → plot.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Sequential, layers

tf.keras.utils.set_random_seed(42)
time = np.linspace(0, 80, 2000)
series = np.sin(time).reshape(-1, 1)
split, window = int(0.8 * len(series)), 30
scaler = MinMaxScaler().fit(series[:split])
scaled = scaler.transform(series).astype("float32")
X, y, target_indices = [], [], []
for i in range(window, len(scaled)):
    X.append(scaled[i-window:i])
    y.append(scaled[i])
    target_indices.append(i)
X, y, target_indices = np.array(X), np.array(y), np.array(target_indices)
mask = target_indices < split
X_train, X_test, y_train, y_test = X[mask], X[~mask], y[mask], y[~mask]
model = Sequential([
    layers.Input(shape=(window, 1)),
    layers.SimpleRNN(32, activation="tanh"), layers.Dense(1)
])
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=15, batch_size=32,
          validation_split=0.1, shuffle=False)
predicted = scaler.inverse_transform(model.predict(X_test, verbose=0))
actual = scaler.inverse_transform(y_test)
print("Test MSE:", mean_squared_error(actual, predicted))
plt.figure(figsize=(12, 4))
plt.plot(time[target_indices[~mask]], actual, label="Actual")
plt.plot(time[target_indices[~mask]], predicted, "--", label="Predicted")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("SimpleRNN: One-step-ahead Forecast")
plt.legend()
plt.tight_layout()
plt.show()
```

**Output:** Test MSE and actual versus predicted values.

**Result:** SimpleRNN predicts sequential values using chronological training and testing periods.

**Evaluation note:** Each prediction uses observed preceding values. This is one-step forecasting, not recursive multi-step forecasting.

## 8. Restricted Boltzmann Machine

**Aim:** Learn MNIST image representations and reconstruct digits using an RBM.

**Algorithm:** Flatten and binarize images → calculate hidden probabilities → sample hidden states → reconstruct visible units → update with contrastive divergence (CD-1) → compare reconstructions.

```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

rng = np.random.default_rng(42)
(x_train, _), (x_test, _) = mnist.load_data()
X = (x_train[:5000].reshape(-1, 784) / 255.0 > 0.5).astype("float32")
X_test = (x_test[:10].reshape(-1, 784) / 255.0 > 0.5).astype("float32")
W = rng.normal(0, 0.01, (784, 64)).astype("float32")
vb = np.zeros(784, dtype="float32")
hb = np.zeros(64, dtype="float32")
lr, batch_size, epochs = 0.05, 100, 10

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -30, 30)))

def hidden_probability(v):
    return sigmoid(v @ W + hb)

def visible_probability(h):
    return sigmoid(h @ W.T + vb)

for epoch in range(epochs):
    shuffled = X[rng.permutation(len(X))]
    errors = []
    for start in range(0, len(shuffled), batch_size):
        v0 = shuffled[start:start + batch_size]
        ph0 = hidden_probability(v0)
        h0 = (rng.random(ph0.shape) < ph0).astype("float32")
        pv1 = visible_probability(h0)
        v1 = (rng.random(pv1.shape) < pv1).astype("float32")
        ph1 = hidden_probability(v1)
        W += lr * (v0.T @ ph0 - v1.T @ ph1) / len(v0)
        vb += lr * np.mean(v0 - v1, axis=0)
        hb += lr * np.mean(ph0 - ph1, axis=0)
        errors.append(np.mean((v0 - pv1) ** 2))
    print(f"Epoch {epoch + 1}: reconstruction MSE = {np.mean(errors):.4f}")
reconstructed = visible_probability(hidden_probability(X_test))
fig, axes = plt.subplots(2, 10, figsize=(15, 4))
for i in range(10):
    axes[0, i].imshow(X_test[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    axes[0, i].axis("off")
    axes[1, i].axis("off")
axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")
plt.tight_layout()
plt.show()
```

**Output:** Reconstruction error and original/reconstructed images. A small RBM can produce blurry results.

**Result:** A single RBM learns representations and reconstructs MNIST images using CD-1. This demonstrates reconstruction rather than conventional image augmentation; it is not a stacked deep RBM.

## 9. LSTM sentiment classification

**Aim:** Predict movie-review sentiment using LSTM.

**Algorithm:** Load tokenized IMDb reviews → pad → embedding → LSTM → sigmoid → train and evaluate → encode user review → classify.

```python
import re
import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

tf.keras.utils.set_random_seed(42)
VOCAB_SIZE, MAX_LENGTH = 10000, 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=VOCAB_SIZE)
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding="post", truncating="post")
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding="post", truncating="post")
model = Sequential([
    layers.Input(shape=(MAX_LENGTH,), dtype="int32"),
    layers.Embedding(VOCAB_SIZE, 64, mask_zero=True),
    layers.LSTM(64), layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test loss: {loss:.4f}; accuracy: {accuracy:.4f}")
word_index = tf.keras.datasets.imdb.get_word_index()
def encode_review(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    sequence = [1]  # Start token; 0 is padding and 2 is unknown.
    for word in words:
        index = word_index.get(word)
        token = index + 3 if index is not None else 2
        sequence.append(token if token < VOCAB_SIZE else 2)
    return pad_sequences([sequence], maxlen=MAX_LENGTH,
                         padding="post", truncating="post")
review = input("Enter a movie review: ")
p = float(model.predict(encode_review(review), verbose=0)[0, 0])
print("Positive probability:", round(p, 4))
print("Sentiment:", "Positive" if p >= 0.5 else "Negative")
```

**Output:** Test metrics and sentiment for the entered review.

**Result:** LSTM performs binary sentiment classification on sequential text.

## 10. Variational Autoencoder

**Aim:** Learn a latent image representation and generate MNIST samples using a VAE.

**Algorithm:** Normalize → encode latent mean and log-variance → reparameterize → decode → minimize reconstruction plus KL loss → reconstruct and generate images.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.utils.set_random_seed(42)
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32").reshape(-1, 784) / 255.0
x_test = x_test.astype("float32").reshape(-1, 784) / 255.0
LATENT_DIM = 2

class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        epsilon = tf.random.normal(tf.shape(mean))
        return mean + tf.exp(0.5 * log_var) * epsilon

encoder_input = tf.keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(encoder_input)
z_mean = layers.Dense(LATENT_DIM, name="z_mean")(x)
z_log_var = layers.Dense(LATENT_DIM, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_input, [z_mean, z_log_var, z])
decoder_input = tf.keras.Input(shape=(LATENT_DIM,))
x = layers.Dense(256, activation="relu")(decoder_input)
decoder_output = layers.Dense(784, activation="sigmoid")(x)
decoder = tf.keras.Model(decoder_input, decoder_output)

class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder, self.decoder = encoder, decoder
        self.total_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_tracker, self.reconstruction_tracker, self.kl_tracker]

    def call(self, inputs, training=False):
        mean, _, z = self.encoder(inputs, training=training)
        return self.decoder(z if training else mean, training=training)

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            mean, log_var, z = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                tf.keras.backend.binary_crossentropy(data, reconstruction), axis=1
            ))
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(
                1 + log_var - tf.square(mean) - tf.exp(log_var), axis=1
            ))
            total_loss = reconstruction_loss + kl_loss
        gradients = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.total_tracker.update_state(total_loss)
        self.reconstruction_tracker.update_state(reconstruction_loss)
        self.kl_tracker.update_state(kl_loss)
        return {metric.name: metric.result() for metric in self.metrics}

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(x_train, epochs=10, batch_size=128)
mean, _, _ = encoder.predict(x_test[:10], verbose=0)
reconstructed = decoder.predict(mean, verbose=0)
generated = decoder.predict(tf.random.normal((10, LATENT_DIM)), verbose=0)
fig, axes = plt.subplots(3, 10, figsize=(15, 5))
for i in range(10):
    axes[0, i].imshow(x_test[i].reshape(28, 28), cmap="gray")
    axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap="gray")
    axes[2, i].imshow(generated[i].reshape(28, 28), cmap="gray")
    for row in range(3):
        axes[row, i].axis("off")
axes[0, 0].set_title("Original")
axes[1, 0].set_title("Reconstructed")
axes[2, 0].set_title("Generated")
plt.tight_layout()
plt.show()
```

**Output:** Reconstruction loss, KL loss, reconstructions and new samples. Generated digits may be blurry or ambiguous.

**Result:** A VAE learns a latent distribution and generates image samples.

## Quick viva reference

| Question | Answer |
|---|---|
| What is supervised learning? | Learning from inputs paired with target labels. |
| Why split data? | To assess performance on examples not used for training. |
| What is precision? | TP / (TP + FP). |
| What is recall? | TP / (TP + FN). |
| What is F1-score? | The harmonic mean of precision and recall. |
| Why normalize inputs? | To improve numerical conditioning and optimization. |
| What is ReLU? | The activation max(0, x). |
| What is softmax? | A transformation producing class probabilities that sum to one. |
| What does pooling do? | Reduces spatial or sequential dimensions by aggregating nearby features. |
| Classification versus detection? | Classification assigns labels; detection also locates objects using bounding boxes. |
| Why use LSTM? | Its gates help retain useful information across sequential steps. |
| What is an RBM? | A probabilistic model with visible and hidden units and no connections within each layer. |
| What is contrastive divergence? | An approximate RBM learning method using short Gibbs sampling chains. |
| What is the VAE reparameterization trick? | Sampling z = mean + exp(0.5 × log-variance) × noise, enabling gradient-based training. |
| Why add KL divergence? | To encourage the latent distribution to match a chosen prior. |

## GitHub upload

Create a repository, select **Add file → Upload files**, upload this file as **README.md**, and commit. GitHub displays it on the repository home page.
