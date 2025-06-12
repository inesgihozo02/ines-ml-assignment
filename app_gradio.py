import pandas as pd
import gradio as gr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the cleaned data
df = pd.read_csv("cleaned_data.csv")

# Use only numeric features and fill missing values
features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].fillna(df[features].median())
y = df['Survived']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Define prediction function
def predict_survival(pclass, age, sibsp, parch, fare):
    input_data = [[pclass, age, sibsp, parch, fare]]
    prediction = model.predict(input_data)[0]
    return "Survived ✅" if prediction == 1 else "Did not survive ❌"

# Gradio UI
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Number(label="Passenger Class (1, 2, or 3)"),
        gr.Number(label="Age"),
        gr.Number(label="Siblings/Spouses Aboard"),
        gr.Number(label="Parents/Children Aboard"),
        gr.Number(label="Fare")
    ],
    outputs="text",
    title="Titanic Survival Predictor",
    description="Enter passenger details to predict survival"
)

interface.launch()
