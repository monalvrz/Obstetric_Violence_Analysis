# import necessary libraries
from tensorflow.keras.models import load_model
from flask import (Flask,
                    render_template,
                    request,
                    redirect,
                    url_for,
                    send_static_file)
from .neural_network import (get_ML_dataset,
                                DataFrame_X_y_split,
                                Clustered_NN_Classifier)
                                
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
NN_threshold_dict = {'P10_8_1': 0.9973000288009644,
                    'P10_8_2': 0.9991999864578247,
                    'P10_8_3': 0.8960999846458435,
                    'P10_8_4': 0.9894999861717224,
                    'P10_8_5': 0.9994000196456909,
                    'P10_8_6': 0.9973000288009644,
                    'P10_8_7': 0.9060999751091003,
                    'P10_8_8': 0.9829999804496765,
                    'P10_8_9': 0.9750999808311462,
                    'P10_8_10': 0.9993000030517578,
                    'P10_8_11': 0.9973000288009644}

#################################################
# Saved Model dictionary
#################################################
NN_model_dict = {}
for key in NN_threshold_dict.keys():
    NN_model_dict[key] = load_model(f'NN_Saved_Models/{key}_model.h5')

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
        user_inputs["NOM_ENT"] = request.form["NOM_ENT"]
        user_inputs["DOMINIO"] = request.form["DOMINIO"]
        user_inputs["EDAD"] = request.form["EDAD"]
        user_inputs["NIV"] = request.form["NIV"]
        user_inputs["P1_1"] = request.form["P1_1"]
        user_inputs["P1_2"] = request.form["P1_2"]
        user_inputs["P1_2_A"] = request.form["P1_2_A"]
        user_inputs["P1_3"] = request.form["P1_3"]
        user_inputs["P1_4_1"] = request.form["P1_4_1"]
        user_inputs["P1_4_2"] = request.form["P1_4_2"]
        user_inputs["P1_4_3"] = request.form["P1_4_3"]
        user_inputs["P1_4_4"] = request.form["P1_4_4"]
        user_inputs["P1_4_5"] = request.form["P1_4_5"]
        user_inputs["P1_4_6"] = request.form["P1_4_6"]
        user_inputs["P1_4_7"] = request.form["P1_4_7"]
        user_inputs["P1_4_8"] = request.form["P1_4_8"]
        user_inputs["P1_4_9"] = request.form["P1_4_9"]
        user_inputs["P1_5"] = request.form["P1_5"]
        user_inputs["P1_6"] = request.form["P1_6"]
        user_inputs["P1_7"] = request.form["P1_7"]
        user_inputs["P1_8"] = request.form["P1_8"]
        user_inputs["P1_9"] = request.form["P1_9"]
        user_inputs["P1_10_1"] = request.form["P1_10_1"]
        user_inputs["P1_10_2"] = request.form["P1_10_2"]
        user_inputs["P1_10_3"] = request.form["P1_10_3"]
        user_inputs["P1_10_4"] = request.form["P1_10_4"]
        user_inputs["P2_5"] = request.form["P2_5"]
        user_inputs["P2_6"] = request.form["P2_6"]
        user_inputs["P2_8"] = request.form["P2_8"]
        user_inputs["P2_9"] = request.form["P2_9"]
        user_inputs["P2_10"] = request.form["P2_10"]
        user_inputs["P2_11"] = request.form["P2_11"]
        user_inputs["P2_12"] = request.form["P2_12"]
        user_inputs["P2_13"] = request.form["P2_13"]
        user_inputs["P2_14"] = request.form["P2_14"]
        user_inputs["P2_15"] = request.form["P2_15"]
        user_inputs["P2_16"] = request.form["P2_16"]
        user_inputs["P3_1"] = request.form["P3_1"]
        user_inputs["P3_2"] = request.form["P3_2"]
        user_inputs["P3_3"] = request.form["P3_3"]
        user_inputs["P3_4"] = request.form["P3_4"]
        user_inputs["P3_5"] = request.form["P3_5"]
        user_inputs["P3_6"] = request.form["P3_6"]
        user_inputs["P3_7"] = request.form["P3_7"]
        user_inputs["P3_8"] = request.form["P3_8"]
        user_inputs["P4AB_1"] = request.form["P4AB_1"]
        user_inputs["P4AB_2"] = request.form["P4AB_2"]
        user_inputs["P4A_1"] = request.form["P4A_1"]
        user_inputs["P4A_2"] = request.form["P4A_2"]
        user_inputs["P4B_1"] = request.form["P4B_1"]
        user_inputs["P4B_2"] = request.form["P4B_2"]
        user_inputs["P4BC_1"] = request.form["P4BC_1"]
        user_inputs["P4BC_2"] = request.form["P4BC_2"]
        user_inputs["P4C_1"] = request.form["P4C_1"]
        user_inputs["P4BC_3"] = request.form["P4BC_3"]
        user_inputs["P4BC_4"] = request.form["P4BC_4"]
        user_inputs["P4BC_5"] = request.form["P4BC_5"]
        user_inputs["P4_1"] = request.form["P4_1"]
        user_inputs["P4_2"] = request.form["P4_2"]
        user_inputs["P4_2_1"] = request.form["P4_2_1"]
        user_inputs["P4_3"] = request.form["P4_3"]
        user_inputs["P4_4"] = request.form["P4_4"]
        user_inputs["P4_5_AB"] = request.form["P4_5_AB"]
        user_inputs["P4_5_1_AB"] = request.form["P4_5_1_AB"]
        user_inputs["P4_6_AB"] = request.form["P4_6_AB"]
        user_inputs["P4_7_AB"] = request.form["P4_7_AB"]
        user_inputs["P4_8_1"] = request.form["P4_8_1"]
        user_inputs["P4_8_2"] = request.form["P4_8_2"]
        user_inputs["P4_8_3"] = request.form["P4_8_3"]
        user_inputs["P4_8_4"] = request.form["P4_8_4"]
        user_inputs["P4_8_5"] = request.form["P4_8_5"]
        user_inputs["P4_8_6"] = request.form["P4_8_6"]
        user_inputs["P4_8_7"] = request.form["P4_8_7"]
        user_inputs["P4_9_1"] = request.form["P4_9_1"]
        user_inputs["P4_9_2"] = request.form["P4_9_2"]
        user_inputs["P4_10_2_1"] = request.form["P4_10_2_1"]
        user_inputs["P4_10_2_2"] = request.form["P4_10_2_2"]
        user_inputs["P4_10_2_3"] = request.form["P4_10_2_3"]
        user_inputs["P4_9_3"] = request.form["P4_9_3"]
        user_inputs["P4_10_3_1"] = request.form["P4_10_3_1"]
        user_inputs["P4_10_3_2"] = request.form["P4_10_3_2"]
        user_inputs["P4_10_3_3"] = request.form["P4_10_3_3"]
        user_inputs["P4_9_4"] = request.form["P4_9_4"]
        user_inputs["P4_9_5"] = request.form["P4_9_5"]
        user_inputs["P4_9_6"] = request.form["P4_9_6"]
        user_inputs["P4_9_7"] = request.form["P4_9_7"]
        user_inputs["P4_11"] = request.form["P4_11"]
        user_inputs["P4_12_1"] = request.form["P4_12_1"]
        user_inputs["P4_12_2"] = request.form["P4_12_2"]
        user_inputs["P4_12_3"] = request.form["P4_12_3"]
        user_inputs["P4_12_4"] = request.form["P4_12_4"]
        user_inputs["P4_12_5"] = request.form["P4_12_5"]
        user_inputs["P4_12_6"] = request.form["P4_12_6"]
        user_inputs["P4_12_7"] = request.form["P4_12_7"]
        user_inputs["P4_13_1"] = request.form["P4_13_1"]
        user_inputs["P4_13_2"] = request.form["P4_13_2"]
        user_inputs["P4_13_3"] = request.form["P4_13_3"]
        user_inputs["P4_13_4"] = request.form["P4_13_4"]
        user_inputs["P4_13_5"] = request.form["P4_13_5"]
        user_inputs["P4_13_6"] = request.form["P4_13_6"]
        user_inputs["P4_13_7"] = request.form["P4_13_7"]
        user_inputs["P10_1_1"] = request.form["P10_1_1"]
        user_inputs["P10_1_2"] = request.form["P10_1_2"]
        user_inputs["P10_1_3"] = request.form["P10_1_3"]
        user_inputs["P10_1_4"] = request.form["P10_1_4"]
        user_inputs["P10_1_5"] = request.form["P10_1_5"]
        user_inputs["P10_1_6"] = request.form["P10_1_6"]
        user_inputs["P10_1_7"] = request.form["P10_1_7"]
        user_inputs["P10_1_8"] = request.form["P10_1_8"]
        user_inputs["P10_1_9"] = request.form["P10_1_9"]
        user_inputs["P10_2"] = request.form["P10_2"]
        user_inputs["P10_3"] = request.form["P10_3"]
        user_inputs["P10_4_1"] = request.form["P10_4_1"]
        user_inputs["P10_4_2"] = request.form["P10_4_2"]
        user_inputs["P10_4_3"] = request.form["P10_4_3"]
        user_inputs["P10_5_01"] = request.form["P10_5_01"]
        user_inputs["P10_5_02"] = request.form["P10_5_02"]
        user_inputs["P10_5_03"] = request.form["P10_5_03"]
        user_inputs["P10_5_04"] = request.form["P10_5_04"]
        user_inputs["P10_5_05"] = request.form["P10_5_05"]
        user_inputs["P10_5_06"] = request.form["P10_5_06"]
        user_inputs["P10_5_07"] = request.form["P10_5_07"]
        user_inputs["P10_5_08"] = request.form["P10_5_08"]
        user_inputs["P10_5_09"] = request.form["P10_5_09"]
        user_inputs["P10_5_10"] = request.form["P10_5_10"]
        user_inputs["P10_5_11"] = request.form["P10_5_11"]
        user_inputs["P10_6ANIO"] = request.form["P10_6ANIO"]
        user_inputs["P10_6MES"] = request.form["P10_6MES"]
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
    ml_result = Clustered_NN_Classifier(NN_model_dict[key],NN_threshold_dict[key],ml_dataset[key]['X'])
    return render_template("ml_results.html", user_inputs=user_inputs, NN_result = ml_result)

if __name__ == "__main__":
    app.run()
