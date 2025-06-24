# import pickle
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder

# def main():
#     # Load the trained CatBoost model from pickle file
#     model_path = "catboost_model (3).pkl"
#     with open(model_path, "rb") as f:
#         model = pickle.load(f)

#     # Load original dataset (used for building encoders)
#     data_path = "students_adaptability_level_online_education.csv"
#     df = pd.read_csv(data_path)

#     # Identify categorical columns (object dtype), excluding target
#     object_cols = df.select_dtypes(include=['object']).columns.tolist()
#     target_col = 'Adaptivity Level'
#     if target_col in object_cols:
#         object_cols.remove(target_col)

#     # Build a LabelEncoder for each categorical column based on training data
#     encoders = {}
#     for col in object_cols:
#         le = LabelEncoder()
#         le.fit(df[col].astype(str))  # Ensure all values are string type
#         encoders[col] = le

#     # Create sample input DataFrame with your example data
#     sample_data = pd.DataFrame({
#         'Gender': ['Boy'],
#         'Age': ['21-25'],
#         'Education Level': ['University'],
#         'Institution Type': ['Non Government'],
#         'IT Student': ['Yes'],
#         'Location': ['Yes'],
#         'Load-shedding': ['Low'],
#         'Financial Condition': ['Rich'],
#         'Internet Type': ['Wifi'],
#         'Network Type': ['4G'],
#         'Class Duration': ['1-3'],
#         'Self Lms': ['Yes'],
#         'Device': ['Computer']
#     })

#     # Encode sample input using the same encoders as training data
#     encoded_sample = sample_data.copy()
#     for col in encoded_sample.columns:
#         if col in encoders:
#             # Handle unknown values gracefully by mapping them to a default value (optional)
#             # Here, we assume sample values are known, else transform will raise an error
#             encoded_sample[col] = encoders[col].transform(encoded_sample[col].astype(str))

#     # Make prediction using the encoded sample
#     prediction = model.predict(encoded_sample)

#     # Decode predicted label back to original category name
#     target_encoder = LabelEncoder()
#     target_encoder.fit(df[target_col].astype(str))
#     predicted_label = target_encoder.inverse_transform(prediction)

#     print("Predicted Adaptivity Level:", predicted_label[0])


# if __name__ == "__main__":
#     main()


from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load model
with open("catboost_model (3).pkl", "rb") as f:
    model = pickle.load(f)

# Load original dataset
df = pd.read_csv("students_adaptability_level_online_education.csv")

# Set target and categorical feature columns
target_col = 'Adaptivity Level'
object_cols = df.select_dtypes(include=['object']).columns.tolist()
if target_col in object_cols:
    object_cols.remove(target_col)

# Prepare encoders
encoders = {}
for col in object_cols:
    le = LabelEncoder()
    le.fit(df[col].astype(str))
    encoders[col] = le

# Target encoder
target_encoder = LabelEncoder()
target_encoder.fit(df[target_col].astype(str))

# Dropdown options for the quiz
dropdown_options = {
    'Gender': ['Boy', 'Girl'],
    'Age': ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30"],
    'Education Level': ['School', 'College', 'University'],
    'Institution Type': ['Government', 'Non Government'],
    'IT Student': ['Yes', 'No'],
    'Location': ['Yes', 'No'],
    'Load-shedding': ['Low', 'High'],
    'Financial Condition': ['Poor', 'Mid', 'Rich'],
    'Internet Type': ['Mobile Data', 'Wifi'],
    'Network Type': ['2G', '3G', '4G'],
    'Class Duration': ['0', '1-3', '3-6'],
    'Self Lms': ['Yes', 'No'],
    'Device': ['Mobile', 'Computer', 'Tab']
}

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/quiz")
def quiz():
    return render_template("quiz.html", fields=dropdown_options.keys(), options=dropdown_options)

@app.route("/predict", methods=["POST"])
def predict():
    user_input = {field: [request.form[field]] for field in dropdown_options}
    sample_df = pd.DataFrame(user_input)

    # Encode input using the trained label encoders
    for col in sample_df.columns:
        if col in encoders:
            sample_df[col] = encoders[col].transform(sample_df[col].astype(str))

    # Predict
    prediction = model.predict(sample_df)
    result = target_encoder.inverse_transform(prediction)[0].strip().capitalize()


    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
