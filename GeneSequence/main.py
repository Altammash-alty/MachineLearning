
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report,
                              confusion_matrix, accuracy_score)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Embedding, Bidirectional,
                                      LSTM, Dense, Dropout,
                                      GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(42)
np.random.seed(42)

# ─────────────────────────────────────────────
# 1.  LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  Gene Sequence LSTM Classifier")
print("=" * 60)

def load_split(path):
    df = pd.read_csv(path)
    df = df[["NucleotideSequence", "GeneType"]].dropna()
    # Strip leading '<' and trailing '>' if present
    df["NucleotideSequence"] = (df["NucleotideSequence"]
                                 .str.strip()
                                 .str.lstrip("<")
                                 .str.rstrip(">"))
    return df

train_df = load_split("/tmp/data/train.csv")
val_df   = load_split("/tmp/data/validation.csv")
test_df  = load_split("/tmp/data/test.csv")

print(f"\n[Split sizes]  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
print(f"\n[Class distribution (train)]:")
print(train_df["GeneType"].value_counts())

# ─────────────────────────────────────────────
# 2.  LABEL ENCODING
# ─────────────────────────────────────────────
le = LabelEncoder()
le.fit(train_df["GeneType"])

CLASSES   = list(le.classes_)
N_CLASSES = len(CLASSES)
print(f"\n[Classes] ({N_CLASSES}): {CLASSES}")

y_train = le.transform(train_df["GeneType"])
y_val   = le.transform(val_df["GeneType"])
y_test  = le.transform(test_df["GeneType"])

# one-hot for categorical_crossentropy
y_train_ohe = to_categorical(y_train, N_CLASSES)
y_val_ohe   = to_categorical(y_val,   N_CLASSES)
y_test_ohe  = to_categorical(y_test,  N_CLASSES)

# ─────────────────────────────────────────────
# 3.  TOKENISATION  (character-level)
# ─────────────────────────────────────────────
# Vocabulary: A T G C N (+ padding token 0)
VOCAB = {c: i + 1 for i, c in enumerate("ATGCN")}
PAD_IDX = 0

def encode_seq(seq: str) -> list:
    """Map each nucleotide to an integer; unknown chars → 5 (N-index)."""
    return [VOCAB.get(c.upper(), 5) for c in seq]

MAX_LEN = 500   # cap at 500 to keep memory manageable (~90th-pct coverage)

def encode_and_pad(df, max_len=MAX_LEN):
    seqs = [encode_seq(s) for s in df["NucleotideSequence"]]
    return pad_sequences(seqs, maxlen=max_len, padding="post",
                         truncating="post", value=PAD_IDX)

X_train = encode_and_pad(train_df)
X_val   = encode_and_pad(val_df)
X_test  = encode_and_pad(test_df)

VOCAB_SIZE = len(VOCAB) + 1   # +1 for padding token

print(f"\n[Sequences]  X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")
print(f"[Vocab size] {VOCAB_SIZE}  |  Max sequence length: {MAX_LEN}")

# ─────────────────────────────────────────────
# 4.  CLASS WEIGHTS  (dataset is imbalanced)
# ─────────────────────────────────────────────
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight("balanced",
                           classes=np.unique(y_train),
                           y=y_train)
class_weight_dict = {i: w for i, w in enumerate(cw)}
print(f"\n[Class Weights]: {class_weight_dict}")

# ─────────────────────────────────────────────
# 5.  MODEL DEFINITION
# ─────────────────────────────────────────────
EMBED_DIM = 16    # small embedding; vocabulary only has 6 tokens

def build_model(vocab_size, embed_dim, max_len, n_classes):
    inp = Input(shape=(max_len,), name="sequence_input")

    # Character embedding
    x = Embedding(vocab_size, embed_dim,
                  mask_zero=True, name="char_embedding")(inp)
    x = Dropout(0.2)(x)

    # Bi-LSTM layer 1
    x = Bidirectional(LSTM(128, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.1),
                       name="bilstm_1")(x)

    # Bi-LSTM layer 2
    x = Bidirectional(LSTM(64, return_sequences=True,
                           dropout=0.2, recurrent_dropout=0.1),
                       name="bilstm_2")(x)

    # Bi-LSTM layer 3 (final)
    lstm_out = Bidirectional(LSTM(32, return_sequences=True,
                                  dropout=0.1),
                              name="bilstm_3")(x)

    # Global average pool captures overall sequence representation
    pooled = GlobalAveragePooling1D(name="gap")(lstm_out)

    # Classifier head
    x = Dense(128, activation="relu")(pooled)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    out = Dense(n_classes, activation="softmax", name="output")(x)

    model = Model(inp, out, name="GeneSeq_BiLSTM")
    return model

model = build_model(VOCAB_SIZE, EMBED_DIM, MAX_LEN, N_CLASSES)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ─────────────────────────────────────────────
# 6.  CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=10,
                  restore_best_weights=True, mode="max", verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5,
                      min_lr=1e-6, verbose=1),
    ModelCheckpoint("/tmp/gene_lstm_best.keras",
                    monitor="val_accuracy", save_best_only=True,
                    mode="max", verbose=0),
]

# ─────────────────────────────────────────────
# 7.  TRAINING
# ─────────────────────────────────────────────
print("\n[Training] Starting …")
history = model.fit(
    X_train, y_train_ohe,
    validation_data=(X_val, y_val_ohe),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight_dict,
    verbose=1,
)

# ─────────────────────────────────────────────
# 8.  EVALUATION
# ─────────────────────────────────────────────
y_pred_proba = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)

acc = accuracy_score(y_test, y_pred)
print(f"\n[Test Accuracy]  {acc * 100:.2f}%")
print("\n[Classification Report]")
print(classification_report(y_test, y_pred,
                             target_names=CLASSES,
                             zero_division=0))

# ─────────────────────────────────────────────
# 9.  PLOTS
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("Gene Sequence BiLSTM Classifier", fontsize=16, fontweight="bold")

# --- (a) Training history: accuracy ---
ax = axes[0]
ax.plot(history.history["accuracy"],     label="Train Accuracy")
ax.plot(history.history["val_accuracy"], label="Val Accuracy")
ax.set_title("Accuracy over Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.legend(); ax.grid(True, alpha=0.3)

# --- (b) Training history: loss ---
ax = axes[1]
ax.plot(history.history["loss"],     label="Train Loss")
ax.plot(history.history["val_loss"], label="Val Loss")
ax.set_title("Loss over Epochs")
ax.set_xlabel("Epoch"); ax.set_ylabel("Categorical Cross-Entropy")
ax.legend(); ax.grid(True, alpha=0.3)

# --- (c) Confusion matrix ---
ax = axes[2]
cm = confusion_matrix(y_test, y_pred)
# Normalise for readability
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, linewidths=0.5)
ax.set_title("Normalised Confusion Matrix (Test)")
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)

plt.tight_layout()
out_path = "/mnt/user-data/outputs/gene_lstm_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n[Saved] Plot → {out_path}")
plt.close()

print("\n[Done] Gene LSTM classification complete.")