import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.utils import to_categorical

# Load and preprocess data
filepath = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df = pd.read_csv(filepath, nrows=10000)
df.drop(columns=['Unnamed: 0'], errors='ignore', inplace=True)

# Normalize features and encode labels
X = MinMaxScaler().fit_transform(df.drop('label', axis=1))
Y = LabelEncoder().fit_transform(df['label'])

# One-hot encode labels for neural network
num_classes = len(np.unique(Y))
Y_one_hot = to_categorical(Y, num_classes=num_classes)

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
Y_train_oh = to_categorical(Y_train, num_classes=num_classes)
Y_test_oh = to_categorical(Y_test, num_classes=num_classes)

# Define models

def build_neural_network(input_dim, num_classes):
    model = Sequential([
        Dense(1024, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.5),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_models():
    results = {}

    # Neural Network
    nn = build_neural_network(X_train.shape[1], num_classes)
    nn.fit(X_train, Y_train_oh, epochs=20, batch_size=128, validation_data=(X_test, Y_test_oh), verbose=1)
    results['Neural Network'] = nn.evaluate(X_test, Y_test_oh, verbose=0)[1]

    # Decision Tree
    dt = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=50)
    dt.fit(X_train, Y_train)
    results['Decision Tree'] = dt.score(X_test, Y_test)

    # Naive Bayes with feature selection
    selector = SelectKBest(f_classif, k=10).fit(X_train, Y_train)
    nb = GaussianNB()
    nb.fit(selector.transform(X_train), Y_train)
    results['Naive Bayes'] = nb.score(selector.transform(X_test), Y_test)

    # SVM
    svm = SVC(kernel='linear', C=1)
    svm.fit(X_train, Y_train)
    results['SVM'] = svm.score(X_test, Y_test)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train, Y_train)
    results['Random Forest'] = rf.score(X_test, Y_test)

    # Gradient Boosting
    gbm = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=42)
    gbm.fit(X_train, Y_train)
    results['Gradient Boosting'] = gbm.score(X_test, Y_test)

    return results, nn, dt, nb, selector, svm, rf, gbm

def plot_conf_matrix(y_true, y_pred, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm.T, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    plt.title(title)
    plt.show()

# Run
start = time.time()
results, nn, dt, nb, selector, svm, rf, gbm = train_and_evaluate_models()
end = time.time()

# Output results
print("\nModel Accuracies:")
for model, acc in results.items():
    print(f"{model}: {acc:.4f}")

print(f"\nTotal training time: {end - start:.2f} seconds")
