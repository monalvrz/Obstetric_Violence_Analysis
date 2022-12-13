# import necessary libraries
from tensorflow.keras.models import load_model
from flask import (Flask,
                    render_template,
                    request,
                    redirect,
                    url_for)
from .neural_network import (get_ML_dataset,
                                DataFrame_X_y_split,
                                NN_Classifier)
                                
#################################################
# Flask Setup
#################################################
app = Flask(__name__)

#################################################
# User Input Dictionary
#################################################
user_inputs = {}

#################################################
# Model Threshold dictionary
#################################################
NN_threshold_dict = {'P10_8_1': 0.03970000147819519,
                    'P10_8_2': 0.0348999984562397,
                    'P10_8_3': 0.010499999858438969,
                    'P10_8_4': 0.04399999976158142,
                    'P10_8_5': 0.0406000018119812,
                    'P10_8_6': 0.04360000044107437,
                    'P10_8_7': 0.03830000013113022,
                    'P10_8_8': 0.05480000004172325,
                    'P10_8_9': 0.0340999998152256,
                    'P10_8_10': 0.02810000069439411,
                    'P10_8_11': 0.03700000047683716}

#################################################
# Saved Model dictionary
#################################################
NN_model_dict = {}
for key in NN_threshold_dict.keys():
    NN_model_dict[key] = load_model(f'deployment/NN_Saved_Models/{key}_model.h5')

# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")

# create favicon.ico route
@app.route('/favicon.ico')
def favicon():
    return url_for('static', filename='image/favicon.ico')

# Access the embeded dashboard results
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# Query the database and send the jsonified results
@app.route("/send", methods=["GET", "POST"])
def send():
    if request.method == "POST":
        # Get the target question for the ML prediction
        user_inputs["Target_Question"] = request.form["Target_Question"]
        # Fill user input dictionary with form data from form.html
        user_inputs["DOMINIO"] = request.form["DOMINIO"]
        user_inputs["EDAD"] = request.form["EDAD"]
        user_inputs["P3_8"] = request.form["P3_8"]
        user_inputs["P10_2"] = request.form["P10_2"]
        user_inputs["P10_7"] = request.form["P10_7"]
        # This will prevent the user input to be dropped from the pre processing
        user_inputs[user_inputs["Target_Question"]] = 1
        return redirect(url_for("ml_results"))

    return render_template("form.html")

@app.route("/result")
def ml_results():
    print(f'Received the following target question: {user_inputs["Target_Question"]}')
    key = user_inputs["Target_Question"]

    ml_dataset = DataFrame_X_y_split(key, get_ML_dataset(user_inputs))
    ml_result = NN_Classifier(NN_model_dict[key],NN_threshold_dict[key],ml_dataset[key]['X'])
    return render_template("ml_results.html", user_inputs=user_inputs, NN_result = ml_result)

if __name__ == "__main__":
    app.run()
