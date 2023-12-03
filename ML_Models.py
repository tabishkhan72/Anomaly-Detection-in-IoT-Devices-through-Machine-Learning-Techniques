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
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout


# Load and preprocess data
filepath = r"C:\Users\tabis\OneDrive\Desktop\SU_Classes\IOT\PROJECT\iot23_combined.csv"
df = pd.read_csv(filepath, nrows=10000)  # Adjust the number of rows if needed
if 'Unnamed: 0' in df.columns:
    df.drop('Unnamed: 0', axis=1, inplace=True)

# Normalize features and convert labels
scaler = MinMaxScaler()
X = scaler.fit_transform(df.drop('label', axis=1))
Y = df['label'].values  # Directly use label column

# Label Encoding
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Split data
X_train, X_test, Y_train_encoded, Y_test_encoded = train_test_split(X, Y_encoded, test_size=0.2, random_state=10)

# Determine the number of unique classes
num_classes = len(label_encoder.classes_)

# One-hot encode the labels for neural network model
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
print(classification_report(Y_test_encoded, label_encoder.transform(Y_pred_nn_names), zero_division=1, target_names=class_names))

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


# In the plot_confusion_matrix function:
def plot_confusion_matrix(model, X_test, Y_test_encoded):
    Y_pred_encoded = model.predict(X_test)
    if len(Y_pred_encoded.shape) > 1 and Y_pred_encoded.shape[1] > 1:
        Y_pred_encoded = np.argmax(Y_pred_encoded, axis=1)  # Only for neural network predictions

    mat = confusion_matrix(Y_test_encoded, Y_pred_encoded)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.show()

def plot_feature_importance(model):
    feature_importances = pd.Series(model.feature_importances_, index=df.columns[:-1])
    feature_importances.nlargest(10).plot(kind='barh')

def plot_multiclass_roc_curve(models, X_test, Y_test, n_classes):
    for model in models:
        y_score = model.predict_proba(X_test)

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        plt.plot(fpr[2], tpr[2], color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc[2])
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic for class {}'.format(i))
        plt.legend(loc="lower right")
        plt.show()

def plot_multiclass_precision_recall_curve(models, X_test, Y_test, n_classes):
    for model in models:
        y_score = model.predict_proba(X_test)

        precision = dict()
        recall = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i))

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="best")
        plt.title("Precision vs. Recall curve")
        plt.show()

plot_confusion_matrix(dt_model, X_test, Y_test_encoded)
plot_confusion_matrix(rf_model, X_test, Y_test_encoded)
plot_confusion_matrix(gbm_model, X_test, Y_test_encoded)

# Feature Importance Plot
plot_feature_importance(rf_model)
plot_feature_importance(gbm_model)

# ROC and Precision-Recall Curves
plot_multiclass_roc_curve([rf_model, gbm_model], X_test, Y_test_one_hot, num_classes)
plot_multiclass_precision_recall_curve([rf_model, gbm_model], X_test, Y_test_one_hot, num_classes)

# Show the plots
plt.show()
