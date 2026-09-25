import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(
    page_title="Titanic ML Playground",
    page_icon="🌳",
    layout="wide"
)

st.title("🚢 Titanic Machine Learning Playground")
st.caption("Streamlit UI based on the Decision Tree notebook: preprocessing, model training, prediction and feature importance.")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader(
    "Upload Titanic CSV",
    type=["csv"],
    help="Upload the titanic.csv used in the notebook."
)

if uploaded_file is None:
    st.info("👈 Upload your Titanic CSV from the sidebar to start.")
    st.markdown("""
    ### Expected notebook-style columns
    `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

    The app can also work with other classification CSV files if you select the target column.
    """)
    st.stop()

df = pd.read_csv(uploaded_file)

st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns")

target = st.sidebar.selectbox(
    "Target column",
    df.columns,
    index=list(df.columns).index("Survived") if "Survived" in df.columns else 0
)

drop_candidates = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
drop_cols = st.sidebar.multiselect(
    "Columns to drop",
    df.columns,
    default=drop_candidates
)

model_name = st.sidebar.selectbox(
    "Model",
    [
        "Decision Tree",
        "Random Forest",
        "KNN",
        "SVM",
        "Logistic Regression",
        "Naive Bayes"
    ]
)

test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 999, 42)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dataset",
    "🧹 Preprocessing",
    "🤖 Train Model",
    "🔮 Prediction",
    "🌳 Tree / Importance"
])

# ---------------- Dataset ----------------
with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.subheader("Missing Values")
    missing = df.isnull().sum().sort_values(ascending=False)
    st.dataframe(
        missing[missing > 0].rename("Missing Count").to_frame(),
        use_container_width=True
    )

    st.subheader("Statistics")
    st.dataframe(df.describe(include="all").T, use_container_width=True)

# ---------------- Preprocessing ----------------
def preprocess(data, target_col, drop_columns):
    work = data.copy()
    work = work.drop(columns=[c for c in drop_columns if c in work.columns], errors="ignore")

    # Target rows with missing labels cannot be used for supervised training.
    work = work.dropna(subset=[target_col])

    X = work.drop(columns=[target_col])
    y = work[target_col]

    # Fill numerical missing values with median.
    num_cols = X.select_dtypes(include=np.number).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())

    # Fill categorical missing values with mode.
    cat_cols = X.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        mode = X[col].mode()
        X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    # Match the notebook: LabelEncode Sex, then get_dummies for remaining categorical columns.
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    X = pd.get_dummies(X, dtype=int)

    # Ensure target is numeric for classification metrics/model display.
    if y.dtype == "object" or str(y.dtype).startswith("category"):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index, name=target_col)

    return X, y

with tab2:
    st.subheader("Preprocessing Pipeline")

    X, y = preprocess(df, target, drop_cols)

    st.write("### Features after preprocessing")
    st.dataframe(X.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Training Features", X.shape[1])
    c2.metric("Samples", X.shape[0])
    c3.metric("Classes", y.nunique())

    st.write("### Applied steps")
    st.markdown("""
    1. Drop selected columns
    2. Remove rows with missing target
    3. Fill numerical missing values with median
    4. Fill categorical missing values with mode
    5. Encode categorical features
    6. Convert categorical features using one-hot encoding
    """)

# ---------------- Model ----------------
def build_model(name):
    if name == "Decision Tree":
        return DecisionTreeClassifier(
            criterion="entropy",
            random_state=random_state
        )
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=10,
            criterion="gini",
            random_state=random_state
        )
    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=7)
    if name == "SVM":
        return SVC()
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    return GaussianNB()

def train_selected_model(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() > 1 else None
    )

    model = build_model(name)

    # Notebook scales KNN/SVM/Logistic Regression/Naive Bayes,
    # while Tree/Random Forest are trained on the original features.
    if name in ["KNN", "SVM", "Logistic Regression", "Naive Bayes"]:
        scaler = StandardScaler()
        X_train_model = scaler.fit_transform(X_train)
        X_test_model = scaler.transform(X_test)
    else:
        scaler = None
        X_train_model = X_train
        X_test_model = X_test

    model.fit(X_train_model, y_train)
    y_pred = model.predict(X_test_model)

    return model, scaler, X_train, X_test, y_train, y_test, y_pred

with tab3:
    st.subheader(f"Train: {model_name}")

    if st.button("🚀 Train Model", type="primary"):
        model, scaler, X_train, X_test, y_train, y_test, y_pred = train_selected_model(X, y, model_name)

        st.session_state["model"] = model
        st.session_state["scaler"] = scaler
        st.session_state["X"] = X
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test
        st.session_state["y_pred"] = y_pred
        st.session_state["model_name"] = model_name

        accuracy = accuracy_score(y_test, y_pred)

        st.success("Model trained successfully!")
        st.metric("Accuracy", f"{accuracy:.2%}")

        st.write("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        st.dataframe(
            pd.DataFrame(cm),
            use_container_width=False
        )

        st.write("### Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)

# ---------------- Prediction ----------------
with tab4:
    st.subheader("Make a Prediction")

    if "model" not in st.session_state:
        st.warning("Train a model first from the **Train Model** tab.")
    else:
        model = st.session_state["model"]
        scaler = st.session_state["scaler"]
        X_model = st.session_state["X"]

        st.write("Enter values for the processed features.")

        input_data = {}
        cols = st.columns(3)

        for i, feature in enumerate(X_model.columns):
            default = float(X_model[feature].median())
            input_data[feature] = cols[i % 3].number_input(
                feature,
                value=default,
                key=f"input_{feature}"
            )

        input_df = pd.DataFrame([input_data], columns=X_model.columns)

        if st.button("🔮 Predict", type="primary"):
            prediction_input = scaler.transform(input_df) if scaler is not None else input_df
            prediction = model.predict(prediction_input)[0]

            st.success(f"Predicted class: **{prediction}**")

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(prediction_input)[0]
                prob_df = pd.DataFrame({
                    "Class": model.classes_,
                    "Probability": probabilities
                })
                st.bar_chart(prob_df.set_index("Class"))

# ---------------- Tree / Importance ----------------
with tab5:
    st.subheader("🌳 Decision Tree Visualization & Feature Importance")

    if "model" not in st.session_state:
        st.warning("Train a model first.")
    else:
        model = st.session_state["model"]
        X_model = st.session_state["X"]

        if isinstance(model, DecisionTreeClassifier):
            fig, ax = plt.subplots(figsize=(18, 9))
            plot_tree(
                model,
                feature_names=X_model.columns,
                class_names=[str(c) for c in model.classes_],
                filled=True,
                max_depth=4,
                fontsize=8,
                ax=ax
            )
            st.pyplot(fig)

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": X_model.columns,
                "Importance": model.feature_importances_
            }).sort_values("Importance", ascending=False)

            st.write("### Which columns does the model give more importance?")
            st.dataframe(importance, use_container_width=True)

            st.bar_chart(
                importance.set_index("Feature").head(15)["Importance"]
            )
        else:
            st.info("Feature importance is available in this UI for Decision Tree and Random Forest models.")

st.sidebar.markdown("---")
st.sidebar.caption("Built from the uploaded Decision Tree notebook.")
