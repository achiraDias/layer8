# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from catboost import CatBoostClassifier




# %%
# Load the training data
train_data = pd.read_csv("train.csv")
valid_data = pd.read_csv("valid.csv")
test_data = pd.read_csv("test.csv")

warnings.filterwarnings("ignore")

# %%
train_data.head()

# %%
valid_data.head()


# %%
test_data.head()

# %%
train_data.head()
missing_matrix = train_data.isnull()
# Create a heatmap using seaborn
plt.figure(figsize=(16, 8))
sns.heatmap(missing_matrix, cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

missing_counts = train_data.isnull().sum()
for column, count in missing_counts.items():
    if(count>0):
      print(f"Column '{column}': {count} missing values")

# %%
valid_data.head()
missing_matrix = valid_data.isnull()
# Create a heatmap using seaborn
plt.figure(figsize=(16, 8))
sns.heatmap(missing_matrix, cbar=False)
plt.title("Missing Values Heatmap")
plt.show()

missing_counts = valid_data.isnull().sum()
for column, count in missing_counts.items():
    if(count>0):
      print(f"Column '{column}': {count} missing values")

# %% [markdown]
# label_2 has missing values

# %%
# Separate features and labels
X_train = train_data.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_train = train_data[['label_1', 'label_2', 'label_3', 'label_4']]
X_valid = valid_data.drop(['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_valid = valid_data[['label_1', 'label_2', 'label_3', 'label_4']]
# X_test = test_data
X_test = test_data.drop(['ID'], axis=1)

# %%
X_train.head()

# %%
Y_train.head()

# %%
X_valid.head()

# %%
Y_valid.head()

# %%
X_test.head()

# %%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# %%
L1 = 'label_1'
L2 = 'label_2'
L3 = 'label_3'
L4 = 'label_4'

print(len(Y_train[L1].unique()))
print(len(Y_train[L2].unique()))
print(len(Y_train[L3].unique()))
print(len(Y_train[L4].unique()))

# %%
def model_accuracy_score(X_train_set, Y_train_set, X_valid_set, Y_valid_set, model):
    model.fit(X_train_set, Y_train_set)
    Y_predicted = model.predict(X_valid_set)
    return accuracy_score(Y_valid_set, Y_predicted)

# %%
def evaluate_models(models, X_train, y_train, X_valid, y_valid):
    model_accuracies = {
        model_name: model_accuracy_score(
            X_train, y_train, X_valid, y_valid, model)
        for model_name, model in models.items()
    }
    # Plot the accuracies
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values())
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.show()

# %%
def hyperparameter_tuning(model, params, X_train_set, Y_train_set):
    # hyper parameter tuning
    model_cv = GridSearchCV(estimator=model, param_grid=params)
    model_cv.fit(X_train_set, Y_train_set)
    return model_cv.best_params_

# %% [markdown]
# ## label_1

# %%
plt.figure(figsize=(18, 6))
sns.countplot(data=Y_train, x=L1, color='lightblue')
plt.xlabel('Speaker', fontsize=12)

# %%
scores_logistic = cross_val_score(LogisticRegression(), X_train, Y_train[L1], cv=3)

# %%
scores_logistic.mean()

# %%
scores_svc = cross_val_score(SVC(C=1000, gamma=0.001), X_train, Y_train[L1], cv=3)

# %%
scores_svc.mean()

# %%
scores_random_forest = cross_val_score(RandomForestClassifier(), X_train, Y_train[L1], cv=3)

# %%
scores_random_forest.mean()

# %% [markdown]
# #### Logistic regression gives better accuracy

# %%
model_accuracy_score(
    X_train_scaled, Y_train[L1],
    X_valid_scaled, Y_valid[L1],
    LogisticRegression()
)

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
X_train_pca

# %%
models = {
    'SVM': SVC(C=1000, gamma=0.001),
    'LogisticRegression': LogisticRegression(),
    'KNN':KNeighborsClassifier(),
    'RandomForest': RandomForestClassifier(),
}

evaluate_models(models, X_train_pca, Y_train[L1], X_valid_pca, Y_valid[L1]) 


# %%
# Parametric tuning

param_grid = {'C': [100, 1000], 'gamma': [0.01, 0.001, 0.0001]}
best_params_svm_l1 = hyperparameter_tuning(
    SVC(), param_grid, X_train_pca, Y_train[L1])
print(best_params_svm_l1)

# %%
model_accuracy_score(
    X_train_pca, Y_train[L1],
    X_valid_pca,
    Y_valid[L1],
    SVC(C=100, gamma=0.001, kernel='rbf'),
)

# %%
best_model_label_1 = SVC(C=100, gamma=0.001, kernel='rbf')

# %%
predicted_label_1 = best_model_label_1.fit(X_train_pca, Y_train[L1]).predict(X_test_pca)

# %%
predicted_label_1.shape

# %% [markdown]
# ## label_2

# %%
train_data[L2].isnull().sum()

# %%
label2_train_data = train_data.copy()
label2_valid_data = valid_data.copy()

# %%
label2_train_data = label2_train_data.dropna(subset=[L2])
label2_valid_data = label2_valid_data.dropna(subset=[L2])

# %%
label2_train_data[L2].isnull().sum()

# %%
label2_train_data.head()

# %%
# Separate features and labels
X_train = label2_train_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_train = label2_train_data[['label_1', 'label_2', 'label_3', 'label_4']]
X_valid = label2_valid_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_valid = label2_valid_data[['label_1', 'label_2', 'label_3', 'label_4']]
# X_test = test_data
X_test = test_data.drop(['ID'], axis=1)

# %%
# Visualize data

plt.figure(figsize=(18, 6))
ax = sns.histplot(data=Y_train, x='label_2', bins=20, kde=False)
plt.xlabel('Speaker Age')

for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=12)

plt.show()

# %%
Y_train[L2].nunique()

# %%
Y_train[L2].value_counts()

# %%
X_train.shape

# %%
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver = 'full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
model_accuracy_score(
    X_train_pca, Y_train[L2],
    X_valid_pca, Y_valid[L2],
    KNeighborsClassifier(n_neighbors=11),
)

# %%
model_accuracy_score(
    X_train_pca, Y_train[L2],
    X_valid_pca, Y_valid[L2],
    SVC(C=1000),
)

# %%
models = {
    'SVM': SVC(C=1000),
    'CatBoost': CatBoostClassifier(loss_function='MultiClass', learning_rate=0.15)
}

evaluate_models(models, X_train_pca, Y_train[L2], X_valid_pca, Y_valid[L2])

# %%
best_model_label_2 = SVC(C=1000)
predicted_label_2 = best_model_label_2.fit(X_train_pca, Y_train[L2]).predict(X_test_pca)

# %% [markdown]
# ## label_3

# %%
# Separate features and labels
X_train = train_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_train = train_data[['label_1', 'label_2', 'label_3', 'label_4']]
X_valid = valid_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_valid = valid_data[['label_1', 'label_2', 'label_3', 'label_4']]
# X_test = test_data
X_test = test_data.drop(['ID'], axis=1)

# %%
ax = sns.countplot(x=Y_train[L3])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='lightblue')
    
plt.xlabel('Speaker Gender')

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
model_accuracy_score(
    X_train_pca, Y_train[L3],
    X_valid_pca,
    Y_valid[L3],
    SVC(),
)

# %%
cross_val_score(SVC(), X_train_pca, Y_train[L3], cv=5).mean()

# %%
predicted_label_3 = SVC().fit(X_train_pca, Y_train[L3]).predict(X_test_pca)

# %% [markdown]
# ## label_4

# %%
# Separate features and labels
X_train = train_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_train = train_data[['label_1', 'label_2', 'label_3', 'label_4']]
X_valid = valid_data.drop(
    ['label_1', 'label_2', 'label_3', 'label_4'], axis=1)
Y_valid = valid_data[['label_1', 'label_2', 'label_3', 'label_4']]
# X_test = test_data
X_test = test_data.drop(['ID'], axis=1)

# %%
ax = sns.countplot(x=y_train[L4])

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha='center', va='bottom', fontsize=9, color='lightblue')
    
plt.xlabel('Speaker Accent')

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

# %%
pca = PCA(n_components=0.95, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
model_accuracy_score(
    X_train_pca, Y_train[L4],
    X_valid_pca,
    Y_valid[L4],
    SVC(class_weight='balanced', C=1000),
)

# %%
cross_val_score(SVC(class_weight='balanced', C=1000), X_train_pca, Y_train[L4], cv=5, scoring='accuracy').mean()

# %%
predicted_label_4 = SVC(class_weight='balanced', C=1000).fit(X_train_pca, Y_train[L4]).predict(X_test_pca)

# %% [markdown]
# ## Generating Output

# %%
output_df = test_data[['ID']]
output_df['label_1'] = predicted_label_1
output_df['label_2'] = predicted_label_2
output_df['label_3'] = predicted_label_3
output_df['label_4'] = predicted_label_4

# %%
output_df.head()

# %%
output_df.to_csv('190137J_output_layer8.csv', index=False)


