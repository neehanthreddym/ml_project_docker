from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

# Create route for homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            math_score = request.form.get('math_score'),
            reading_score = request.form.get('reading_score'),
            writing_score = request.form.get('writing_score'),
        )

        prediction_input_df = data.get_data_as_dataframe()
        print(prediction_input_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(prediction_input_df)
        print(results)

        formatted_result = f"{float(results[0]):.2f}"
        return render_template('home.html', results=formatted_result)

if __name__ == '__main__':
    app.run(host="0.0.0.0")