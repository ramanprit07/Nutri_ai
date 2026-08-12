
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# ============================================================
# NutriSmart AI - Streamlit Application
# Logistic Regression Predictive Model
# Dataset: nutrients_csvfile.csv
# ============================================================

st.set_page_config(
    page_title="NutriSmart AI",
    page_icon="🥗",
    layout="wide",
)

DATA_FILE = "nutrients_csvfile.csv"

RAW_NUMERIC_COLUMNS = [
    "Grams",
    "Calories",
    "Protein",
    "Fat",
    "Sat.Fat",
    "Fiber",
    "Carbs",
]

BASE_FEATURES = [
    "Grams",
    "Calories",
    "Protein",
    "Fat",
    "Sat.Fat",
    "Fiber",
    "Carbs",
]

NORMALIZED_FEATURES = [
    "Calories_100g",
    "Protein_100g",
    "Fat_100g",
    "Sat.Fat_100g",
    "Fiber_100g",
    "Carbs_100g",
]

MODEL_FEATURES = BASE_FEATURES + NORMALIZED_FEATURES


# ============================================================
# Data preparation
# ============================================================

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Clean numeric columns.
    # In this nutrition dataset, "t" is treated as 0 because it
    # represents a trace amount and is not a usable numeric value.
    for col in RAW_NUMERIC_COLUMNS:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
            .replace({"t": 0, "T": 0, "": np.nan, "nan": np.nan})
        )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows without a target category.
    df = df.dropna(subset=["Category"]).copy()

    # Feature engineering: convert nutrient amounts to a 100 g basis.
    # This reduces the effect of different serving sizes.
    for col in [
        "Calories",
        "Protein",
        "Fat",
        "Sat.Fat",
        "Fiber",
        "Carbs",
    ]:
        df[f"{col}_100g"] = np.where(
            df["Grams"] > 0,
            (df[col] / df["Grams"]) * 100,
            np.nan,
        )

    return df


@st.cache_resource
def train_models(df):
    X = df[MODEL_FEATURES]
    y = df["Category"]

    # Stratified split keeps category proportions similar in train/test data.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Baseline: most frequent category.
    baseline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]
    )
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    # Main predictive model: Logistic Regression.
    # Imputation and scaling are inside the pipeline to avoid data leakage.
    logistic_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    multi_class="auto",
                    random_state=42,
                ),
            ),
        ]
    )

    # Tune C using only the training data.
    # search = GridSearchCV(
    #     estimator=logistic_pipeline,
    #     param_grid={"model__C": [0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]},
    #     scoring="f1_weighted",
    #     cv=5,
    #     n_jobs=-1,
    # )
    # search.fit(X_train, y_train)

    # model = search.best_estimator_
    # model_pred = model.predict(X_test)
    # Train Logistic Regression directly
    model = logistic_pipeline.set_params(
        model__C=1.0
    )

    model.fit(X_train, y_train)
    def metrics(actual, predicted):
        return {
            "Accuracy": accuracy_score(actual, predicted),
            "Precision": precision_score(
                actual, predicted, average="weighted", zero_division=0
            ),
            "Recall": recall_score(
                actual, predicted, average="weighted", zero_division=0
            ),
            "F1 Score": f1_score(
                actual, predicted, average="weighted", zero_division=0
            ),
        }

    baseline_metrics = metrics(y_test, baseline_pred)
    model_metrics = metrics(y_test, model_pred)

    results = pd.DataFrame(
        [baseline_metrics, model_metrics],
        index=["Dummy Baseline", "Logistic Regression"],
    ).reset_index().rename(columns={"index": "Model"})

    return {
        "model": model,
        "baseline": baseline,
        "X_test": X_test,
        "y_test": y_test,
        "model_pred": model_pred,
        "baseline_pred": baseline_pred,
        "results": results,
        "best_C": search.best_params_["model__C"],
    }


# ============================================================
# Helper functions
# ============================================================

def build_input_row(grams, calories, protein, fat, sat_fat, fiber, carbs):
    row = {
        "Grams": grams,
        "Calories": calories,
        "Protein": protein,
        "Fat": fat,
        "Sat.Fat": sat_fat,
        "Fiber": fiber,
        "Carbs": carbs,
    }

    for col in [
        "Calories",
        "Protein",
        "Fat",
        "Sat.Fat",
        "Fiber",
        "Carbs",
    ]:
        row[f"{col}_100g"] = (
            (row[col] / grams) * 100 if grams and grams > 0 else np.nan
        )

    return pd.DataFrame([row], columns=MODEL_FEATURES)


def show_metric_cards(results):
    baseline = results.iloc[0]
    logistic = results.iloc[1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Logistic Accuracy", f"{logistic['Accuracy']:.1%}")
    c2.metric("Logistic Precision", f"{logistic['Precision']:.1%}")
    c3.metric("Logistic Recall", f"{logistic['Recall']:.1%}")
    c4.metric("Logistic F1", f"{logistic['F1 Score']:.1%}")

    st.caption(
        f"Baseline accuracy: {baseline['Accuracy']:.1%} | "
        "Metrics are calculated on the held-out test set."
    )


# ============================================================
# Load data and model
# ============================================================

try:
    df = load_and_clean_data(DATA_FILE)
    training = train_models(df)
except FileNotFoundError:
    st.error(
        f"Could not find '{DATA_FILE}'. Put the CSV file in the same folder as app.py."
    )
    st.stop()
except Exception as exc:
    st.error(f"Application setup error: {exc}")
    st.stop()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("# 🥗 NutriSmart AI")
    st.caption("Food Category Prediction using Logistic Regression")

    option = st.radio(
        "Navigation",
        [
            "Home",
            "Nutrient Mapping",
            "Predictive Model",
            "Model Performance",
        ],
    )

    st.divider()
    st.markdown("### Project Workflow")
    st.write("1. Data Cleaning")
    st.write("2. Exploratory Analysis")
    st.write("3. Feature Engineering")
    st.write("4. Logistic Regression")
    st.write("5. Model Evaluation")
    st.write("6. New Food Prediction")


# ============================================================
# HOME
# ============================================================

if option == "Home":
    st.title("🥗 NutriSmart AI")
    st.subheader("Food Nutrition Analysis & Predictive Classification")

    st.markdown(
        """
        This application uses nutritional characteristics to predict the
        **food category** of a food item.

        The predictive model is **Logistic Regression**, with a
        **Dummy Classifier baseline** for comparison.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Food Records", len(df))
    c2.metric("Food Categories", df["Category"].nunique())
    c3.metric("Input Nutrition Features", len(BASE_FEATURES))
    c4.metric("ML Models Compared", 2)

    st.markdown("## Project Objective")
    st.info(
        "Can nutritional characteristics be used to classify food items "
        "into their food categories?"
    )

    st.markdown("## Dataset Overview")

    left, right = st.columns(2)

    with left:
        category_counts = (
            df["Category"]
            .value_counts()
            .reset_index()
        )
        category_counts.columns = ["Category", "Count"]

        fig = px.bar(
            category_counts,
            x="Count",
            y="Category",
            orientation="h",
            title="Food Records by Category",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig2 = px.scatter(
            df,
            x="Calories",
            y="Protein",
            color="Category",
            hover_name="Food",
            title="Calories vs Protein",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("## Data Preparation Decisions")
    st.write(
        "• Commas were removed from numeric values. "
        "• Trace values represented by 't' were treated as 0. "
        "• Numeric columns were converted to numeric data types. "
        "• Missing numeric values are handled by median imputation inside the ML pipeline. "
        "• Nutrient-per-100g features were engineered to account for different serving sizes."
    )


# ============================================================
# NUTRIENT MAPPING
# ============================================================

elif option == "Nutrient Mapping":
    st.title("📊 Nutrient Mapping")

    category = st.selectbox(
        "Select Food Category",
        ["All Categories"] + sorted(df["Category"].unique().tolist()),
    )

    nutrient = st.selectbox(
        "Select Nutrient",
        ["Calories", "Protein", "Fat", "Sat.Fat", "Fiber", "Carbs"],
    )

    if category == "All Categories":
        filtered = df.copy()
    else:
        filtered = df[df["Category"] == category].copy()

    top_foods = filtered.nlargest(10, nutrient)

    fig = px.bar(
        top_foods.sort_values(nutrient),
        x=nutrient,
        y="Food",
        orientation="h",
        color="Category",
        title=f"Top Foods by {nutrient}",
        hover_data=["Grams", "Calories", "Protein", "Fat", "Fiber", "Carbs"],
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Summary Statistics")
    st.dataframe(
        filtered[
            ["Food", "Category", "Grams", "Calories", "Protein", "Fat", "Fiber", "Carbs"]
        ].sort_values(nutrient, ascending=False).head(10),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# PREDICTIVE MODEL
# ============================================================

elif option == "Predictive Model":
    st.title("🤖 Predictive Food Category Model")
    st.subheader("Logistic Regression")

    st.markdown(
        """
        Enter nutritional information for a new food item. The trained
        Logistic Regression model will predict the most likely food category.
        """
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            grams = st.number_input(
                "Serving Weight (grams)",
                min_value=1.0,
                value=100.0,
                step=1.0,
            )
            calories = st.number_input(
                "Calories",
                min_value=0.0,
                value=200.0,
                step=1.0,
            )
            protein = st.number_input(
                "Protein (g)",
                min_value=0.0,
                value=10.0,
                step=0.1,
            )

        with col2:
            fat = st.number_input(
                "Fat (g)",
                min_value=0.0,
                value=5.0,
                step=0.1,
            )
            sat_fat = st.number_input(
                "Saturated Fat (g)",
                min_value=0.0,
                value=2.0,
                step=0.1,
            )

        with col3:
            fiber = st.number_input(
                "Fiber (g)",
                min_value=0.0,
                value=2.0,
                step=0.1,
            )
            carbs = st.number_input(
                "Carbohydrates (g)",
                min_value=0.0,
                value=25.0,
                step=0.1,
            )

        submitted = st.form_submit_button(
            "🔮 Predict Food Category",
            use_container_width=True,
        )

    if submitted:
        input_df = build_input_row(
            grams,
            calories,
            protein,
            fat,
            sat_fat,
            fiber,
            carbs,
        )

        model = training["model"]
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        classes = model.named_steps["model"].classes_

        probability_table = pd.DataFrame(
            {
                "Category": classes,
                "Probability": probabilities,
            }
        ).sort_values("Probability", ascending=False)

        top_probability = probability_table.iloc[0]["Probability"]

        st.success(f"Predicted Food Category: **{prediction}**")
        st.metric("Prediction Probability", f"{top_probability:.1%}")

        st.markdown("### Prediction Probabilities")

        prob_chart = px.bar(
            probability_table.head(8).sort_values("Probability"),
            x="Probability",
            y="Category",
            orientation="h",
            title="Top Category Probabilities",
        )
        prob_chart.update_xaxes(tickformat=".0%")
        st.plotly_chart(prob_chart, use_container_width=True)

        st.caption(
            "The probability represents the Logistic Regression model's "
            "estimated class probability; it is not a guarantee of the true category."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif option == "Model Performance":
    st.title("📈 Model Performance & Evaluation")

    results = training["results"]

    st.markdown("## Test-Set Performance")
    show_metric_cards(results)

    st.dataframe(
        results.style.format(
            {
                "Accuracy": "{:.2%}",
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    comparison_long = results.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 Score"],
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        comparison_long,
        x="Metric",
        y="Score",
        color="Model",
        barmode="group",
        title="Baseline vs Logistic Regression",
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("## Confusion Matrix — Logistic Regression")

    labels = sorted(df["Category"].unique())
    cm = confusion_matrix(
        training["y_test"],
        training["model_pred"],
        labels=labels,
    )

    cm_fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        aspect="auto",
        labels={
            "x": "Predicted Category",
            "y": "Actual Category",
            "color": "Count",
        },
        title="Confusion Matrix",
    )
    cm_fig.update_xaxes(tickangle=45)
    st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("## Model Configuration")
    st.write(
        f"Best Logistic Regression C selected by 5-fold cross-validation: "
        f"**{training['best_C']}**"
    )
    st.write(
        "The data was split using an 80/20 stratified train/test split "
        "with random_state=42. Imputation and scaling were fitted inside "
        "the training pipeline to reduce data leakage."
    )

    st.markdown("## Logistic Regression Feature Importance")

    logistic_model = training["model"].named_steps["model"]
    coefficients = np.abs(logistic_model.coef_).mean(axis=0)

    importance = (
        pd.DataFrame(
            {
                "Feature": MODEL_FEATURES,
                "Importance": coefficients,
            }
        )
        .sort_values("Importance", ascending=True)
    )

    imp_fig = px.bar(
        importance,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Average Absolute Logistic Regression Coefficient",
    )
    st.plotly_chart(imp_fig, use_container_width=True)

    st.markdown("### Interpretation")
    st.write(
        "Larger absolute coefficients indicate features that contributed more "
        "strongly to separating the food categories in the fitted multinomial "
        "Logistic Regression model. This should be interpreted as model "
        "association, not causation."
    )

# ============================================================
# Footer
# ============================================================

st.divider()
st.caption(
    "NutriSmart AI | Educational machine-learning project | "
    "Predictions are for analytical demonstration and should not be treated as medical advice."
)
