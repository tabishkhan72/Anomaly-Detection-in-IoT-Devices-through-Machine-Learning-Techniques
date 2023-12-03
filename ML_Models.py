import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
import seaborn as sns


# Load and preprocess data
filepath = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df = pd.read_csv(filepath, nrows=10000)  # Adjust the number of rows if needed
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Normalize features and convert labels
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('label', axis=1))
Y = df['label'].values  # Directly use label column

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Label Encoding (Optional)
label_encoder = LabelEncoder()
Y_train_encoded = label_encoder.fit_transform(Y_train)
Y_test_encoded = label_encoder.transform(Y_test)
# Determine the number of unique classes

# num_classes = len(np.unique(Y_train_encoded))
num_classes = len(label_encoder.classes_)
# One-hot encode the labels
Y_train_one_hot = to_categorical(Y_train_encoded, num_classes=num_classes)
Y_test_one_hot = to_categorical(Y_test_encoded, num_classes=num_classes)

# Deep Neural Network Model
def train_optimized_neural_network(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    # Update the number of neurons in the last layer to match the number of classes
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_test, Y_test), verbose=1)

    return model

# Decision Tree Model
def train_optimized_decision_tree(X_train, Y_train):
    DT = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=50)
    DT.fit(X_train, Y_train)
    return DT

# Optimized Gaussian Naive Bayes Model
def train_optimized_gaussian_nb(X_train, Y_train):
    selector = SelectKBest(f_classif, k=10)
    X_new = selector.fit_transform(X_train, Y_train)
    clf = GaussianNB()
    clf.fit(X_new, Y_train)
    return clf, selector

# Support Vector Machine Model
def train_optimized_svm(X_train, Y_train, use_sgd=False):
    if use_sgd:
        # Using Stochastic Gradient Descent
        svm_model = SGDClassifier(loss='hinge', penalty='l2', alpha=1/(10*X_train.shape[0]), verbose=0)
    else:
        # Using traditional SVM with linear kernel
        svm_model = SVC(C=1, kernel='linear', cache_size=1500, verbose=False)
    
    svm_model.fit(X_train, Y_train)
    return svm_model

# Random Forest Model
def train_optimized_random_forest(X_train, Y_train_label):
    RF = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    RF.fit(X_train, Y_train_label)
    return RF

# Gradient Boosting Model
def train_optimized_gbm(X_train, Y_train_label):
    GBM = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=42)
    GBM.fit(X_train, Y_train_label)
    return GBM

# Train and evaluate models
start_time = time.time()
# Train models with encoded labels
nn_model = train_optimized_neural_network(X_train, Y_train_one_hot, X_test, Y_test_one_hot)
dt_model = train_optimized_decision_tree(X_train, Y_train_encoded)
nb_model, selector = train_optimized_gaussian_nb(X_train, Y_train_encoded)
svm_model = train_optimized_svm(X_train, Y_train_encoded)
rf_model = train_optimized_random_forest(X_train, Y_train_encoded)
gbm_model = train_optimized_gbm(X_train, Y_train_encoded)

end_time = time.time()

# Transform the test set with the same SelectKBest instance
X_test_transformed = selector.transform(X_test)

# Evaluate Models
print("Neural Network Accuracy:", nn_model.evaluate(X_test,Y_test_one_hot)[1])
print("Decision Tree Accuracy:", dt_model.score(X_test, Y_test_encoded))
print("Gaussian Naive Bayes Accuracy:", nb_model.score(X_test_transformed, Y_test_encoded))
print("SVM Accuracy:", svm_model.score(X_test, Y_test_encoded))
print("Random Forest Accuracy:", rf_model.score(X_test, Y_test_encoded))
print("GBM Accuracy:", gbm_model.score(X_test, Y_test_encoded))

# Print time taken
print("Time taken: {:.2f} seconds".format(end_time - start_time))

Y_pred_nn_encoded = nn_model.predict(X_test)
Y_pred_nn = np.argmax(Y_pred_nn_encoded, axis=1)
Y_pred_dt = label_encoder.inverse_transform(dt_model.predict(X_test))
Y_pred_nb = label_encoder.inverse_transform(nb_model.predict(X_test_transformed))
Y_pred_svm = label_encoder.inverse_transform(svm_model.predict(X_test))
Y_pred_rf = label_encoder.inverse_transform(rf_model.predict(X_test))
Y_pred_gbm = label_encoder.inverse_transform(gbm_model.predict(X_test))

class_names = label_encoder.classes_
Y_pred_nn_names = [class_names[i] for i in Y_pred_nn]

# Generate classification reports with original labels
print("Neural Network Report:")
# print(classification_report(Y_test_encoded, Y_pred_nn))
print(classification_report(label_encoder.transform(Y_test), label_encoder.transform(Y_pred_nn_names),zero_division=1, target_names=class_names))

print("Decision Tree Classification Report:")
print(classification_report(label_encoder.inverse_transform(Y_test_encoded), Y_pred_dt))

print("Gaussian Naive Bayes Classification Report:")
print(classification_report(label_encoder.inverse_transform(Y_test_encoded), Y_pred_nb))

print("SVM Classification Report:")
print(classification_report(label_encoder.inverse_transform(Y_test_encoded), Y_pred_svm))

print("Random Forest Classification Report:")
print(classification_report(label_encoder.inverse_transform(Y_test_encoded), Y_pred_rf))

print("Gradient Boosting Machine Classification Report:")
print(classification_report(label_encoder.inverse_transform(Y_test_encoded), Y_pred_gbm))
