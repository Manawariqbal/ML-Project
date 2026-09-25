# Machine Learning Playground

An interactive **Streamlit-based Machine Learning application** for exploring the any dataset, preprocessing data, training classification models, evaluating their performance, making predictions, and visualizing feature importance.

The application is based on the workflow used in the accompanying Decision Tree notebook and provides a user-friendly UI around the machine learning pipeline.

---

## ✨ Features

- 📂 Upload a CSV dataset directly from the Streamlit sidebar
- 📊 Explore dataset rows, columns, statistics, and missing values
- 🧹 Perform basic data preprocessing
- 🔤 Encode categorical variables
- 🤖 Train multiple classification algorithms
- 📈 Evaluate models using accuracy, confusion matrix, and classification report
- 🔮 Make predictions using an interactive form
- 🌳 Visualize the Decision Tree
- ⭐ View feature importance for Decision Tree and Random Forest
- 📊 Compare the contribution of individual features

---

## 🤖 Supported Machine Learning Models

The application supports the following classification algorithms:

1. **Decision Tree Classifier**
   - Uses `entropy` as the splitting criterion.
2. **Random Forest Classifier**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Logistic Regression**
6. **Gaussian Naive Bayes**

---

## 📁 Project Structure

```text
titanic-ml-streamlit/
│
├── app.py
├── requirements.txt
├── README.md
│
└── titanic.csv              # Dataset supplied by the user
```

---

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd titanic-ml-streamlit
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Start the Streamlit application with:

```bash
streamlit run app.py
```

Streamlit will open the application in your browser.

If it does not open automatically, use the local URL displayed in the terminal.

---

## 📂 Dataset

Upload your CSV file using the **Upload Titanic CSV** option in the sidebar.

For the Titanic workflow, the application expects a target column such as:

```text
Survived
```

Typical Titanic features include:

```text
PassengerId
Survived
Pclass
Name
Sex
Age
SibSp
Parch
Ticket
Fare
Cabin
Embarked
```

The target column can also be selected from the sidebar, allowing the application to work with other classification datasets.

---

## 🧹 Data Preprocessing

The application performs the following preprocessing steps:

### 1. Drop unnecessary columns

For a Titanic dataset, columns such as:

```text
PassengerId
Name
Ticket
Cabin
```

can be removed from the sidebar.

### 2. Handle missing target values

Rows where the target value is missing are removed because they cannot be used for supervised training.

### 3. Numerical missing values

Missing numerical values are filled using the **median** of the corresponding feature.

### 4. Categorical missing values

Missing categorical values are filled using the **mode**.

### 5. Categorical encoding

Categorical values are converted into numerical representations.

### 6. One-hot encoding

The processed feature set is converted using:

```python
pd.get_dummies(..., dtype=int)
```

---

## 🌳 Decision Tree

The Decision Tree model is configured with:

```python
DecisionTreeClassifier(
    criterion="entropy",
    random_state=random_state
)
```

The application can visualize the trained tree and display the most important features.

### Entropy

Entropy measures the impurity or uncertainty in a node.

The Decision Tree attempts to select splits that provide a reduction in impurity.

---

## 🌲 Random Forest

Random Forest combines multiple decision trees and aggregates their predictions.

The application uses:

```python
RandomForestClassifier(
    n_estimators=10,
    criterion="gini",
    random_state=random_state
)
```

Random Forest also exposes:

```python
model.feature_importances_
```

which is used by the application to show which features contributed most to the model according to the model's impurity-based feature importance.

---

## 📊 Model Evaluation

After training, the application displays:

### Accuracy

```text
Accuracy = Correct Predictions / Total Predictions
```

### Confusion Matrix

The confusion matrix shows the relationship between:

- Actual classes
- Predicted classes

### Classification Report

The report contains metrics such as:

- Precision
- Recall
- F1-score
- Support

---

## 🔮 Interactive Prediction

After training a model, go to the:

```text
Prediction
```

tab.

The application creates input fields for the processed features.

Click:

```text
🔮 Predict
```

to generate the predicted class.

For models that support probability prediction, the application also displays class probabilities.

---

## ⭐ Feature Importance

For models that provide `feature_importances_`, the application displays a table similar to:

| Feature | Importance |
|---|---:|
| Feature A | 0.35 |
| Feature B | 0.21 |
| Feature C | 0.14 |
| Feature D | 0.08 |

This helps answer:

> **Which columns does the model give more importance to?**

For Decision Tree and Random Forest, the application also displays a bar chart of the most important features.

---

## 🌳 Decision Tree Visualization

When the selected model is a Decision Tree, the application visualizes the trained tree using:

```python
plot_tree()
```

The visualization includes:

- Feature used for splitting
- Threshold
- Samples
- Class information
- Tree structure

The displayed tree is limited to a maximum depth for readability.

---

## 🖥️ Application Tabs

The Streamlit application is divided into five main tabs.

### 📊 Dataset

Used for:

- Dataset preview
- Number of rows
- Number of columns
- Missing values
- Statistical summary

### 🧹 Preprocessing

Shows:

- Processed features
- Number of training features
- Number of samples
- Number of classes
- Preprocessing steps

### 🤖 Train Model

Used to:

- Select a model
- Train the model
- View accuracy
- View confusion matrix
- View classification report

### 🔮 Prediction

Used to:

- Enter feature values
- Generate predictions
- View prediction probabilities where supported

### 🌳 Tree / Importance

Used to:

- Visualize Decision Tree
- View feature importance
- View feature-importance chart

---

## ⚙️ Sidebar Configuration

The sidebar provides controls for:

- CSV upload
- Target column
- Columns to drop
- Machine learning model
- Test size
- Random state

Example:

```text
Target Column: Survived
Model: Decision Tree
Test Size: 20%
Random State: 42
```

---

## 📦 Requirements

The project uses the following Python packages:

```text
streamlit
pandas
numpy
scikit-learn
matplotlib
```

Install them with:

```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

```bash
git clone <your-repository-url>
cd titanic-ml-streamlit

python -m venv venv

# Activate environment
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

Then upload your Titanic CSV and start experimenting with the models.

---

## 🎯 Learning Objectives

This project is useful for understanding an end-to-end classification workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Prediction
   ↓
Feature Importance
   ↓
Visualization
```

It can also be used as a teaching/demo application for introductory Machine Learning and Decision Tree concepts.

---

## ⚠️ Notes

- The application is designed primarily for classification datasets.
- The target column must contain usable class labels.
- Feature importance shown by tree-based models is the model's built-in impurity-based importance; it should not automatically be interpreted as causal importance.
- For KNN, SVM, Logistic Regression, and Naive Bayes, the application applies `StandardScaler`.
- Decision Tree and Random Forest do not require feature scaling for this workflow.

---

## 👨‍💻 Author

**Md Manawar Iqbal**

Machine Learning / GenAI Engineer

---

## 📄 License

Add your preferred license here, for example:

```text
MIT License
```

if you choose to release the project under the MIT License.
