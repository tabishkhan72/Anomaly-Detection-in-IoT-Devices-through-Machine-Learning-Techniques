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
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# Load and preprocess data
filepath = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df = pd.read_csv(filepath, nrows=400000)  # Adjust the number of rows if needed
# Check if the column 'Unnamed: 0' exists before dropping
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('label', axis=1))
Y = pd.get_dummies(df['label']).values

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Deep Neural Network Model
def train_optimized_neural_network(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(Y_train.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(X_train, Y_train, epochs=20, batch_size=128, validation_data=(X_test, Y_test), 
              callbacks=[early_stopping], verbose=1)
    return model

# Decision Tree Model
def train_optimized_decision_tree(X_train, Y_train):
    DT = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=50)
    DT.fit(X_train, Y_train)
    return DT

# Gaussian Naive Bayes Model
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
def train_optimized_random_forest(X_train, Y_train):
    RF = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    RF.fit(X_train, Y_train)
    return RF

# Gradient Boosting Model
def train_optimized_gbm(X_train, Y_train):
    GBM = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=5, random_state=42)
    GBM.fit(X_train, Y_train)
    return GBM

# Train and evaluate models
start_time = time.time()
nn_model = train_optimized_neural_network(X_train, Y_train, X_test, Y_test)
dt_model = train_optimized_decision_tree(X_train, np.argmax(Y_train, axis=1))
nb_model = train_optimized_gaussian_nb(X_train, np.argmax(Y_train, axis=1))
svm_model = train_optimized_svm(X_train, np.argmax(Y_train, axis=1))
rf_model = train_optimized_random_forest(X_train, Y_train)
gbm_model = train_optimized_gbm(X_train, Y_train)
end_time = time.time()

# Evaluate Models
print("Neural Network Accuracy:", nn_model.evaluate(X_test, Y_test)[1])
print("Decision Tree Accuracy:", dt_model.score(X_test, np.argmax(Y_test, axis=1)))
print("Gaussian Naive Bayes Accuracy:", nb_model.score(X_test, np.argmax(Y_test, axis=1)))
print("SVM Accuracy:", svm_model.score(X_test, np.argmax(Y_test, axis=1)))
print("Random Forest Accuracy:", rf_model.score(X_test, np.argmax(Y_test, axis=1)))
print("GBM Accuracy:", gbm_model.score(X_test, np.argmax(Y_test, axis=1)))


# Print time taken
print("Time taken: {:.2f} seconds".format(end_time - start_time))

# Print Classification Reports (for models other than NN)
print("Decision Tree Report:")
print(classification_report(np.argmax(Y_test, axis=1), dt_model.predict(X_test)))

print("Gaussian NB Report:")
print(classification_report(np.argmax(Y_test, axis=1), nb_model.predict(X_test)))

print("SVM Report:")
print(classification_report(np.argmax(Y_test, axis=1), svm_model.predict(X_test)))

print("Random Forest Report:")
print(classification_report(np.argmax(Y_test, axis=1), rf_model.predict(X_test)))

print("GBM Report:")
print(classification_report(np.argmax(Y_test, axis=1), gbm_model.predict(X_test)))

# Example usage:
# plot_confusion_matrix(rf_model, X_test, np.argmax(Y_test, axis=1))
# plot_feature_importance(rf_model)
# plot_roc_curve([rf_model, gbm_model], X_test, np.argmax(Y_test, axis=1))
# plot_precision_recall_curve([rf_model, gbm_model], X_test, np.argmax(Y_test, axis=1))

def plot_confusion_matrix(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    mat = confusion_matrix(np.argmax(Y_test, axis=1), np.argmax(Y_pred, axis=1))
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')

def plot_feature_importance(model):
    feature_importances = pd.Series(model.feature_importances_, index=df.columns[:-1])
    feature_importances.nlargest(10).plot(kind='barh')

def plot_roc_curve(models, X_test, Y_test):
    for model in models:
        Y_pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, Y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{type(model).__name__} (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

def plot_precision_recall_curve(models, X_test, Y_test):
    for model in models:
        Y_pred = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(Y_test, Y_pred)
        plt.plot(recall, precision, label=type(model).__name__)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")


