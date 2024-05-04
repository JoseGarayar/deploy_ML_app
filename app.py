from flask import Flask,request,render_template
import numpy as np
import pickle

# Importar los modelos
model = pickle.load(open('pickle_files/model.pkl','rb'))
sc = pickle.load(open('pickle_files/standscaler.pkl','rb'))
ct = pickle.load(open('pickle_files/encoder.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    gender = request.form['gender']
    age = request.form['age']
    annual_income = request.form['annualIncome']
    spending_score = request.form['spendingScore']

    feature_list = ([[gender, age, annual_income, spending_score]])

    transformed_features = ct.transform(feature_list)
    transformed_features[:,2:] = sc.transform(transformed_features[:,2:])
    prediction = model.predict(transformed_features).reshape(1,-1)

    diccionario = {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5"}

    value_predicted = prediction[0][0]
    if value_predicted in diccionario:
        crop = diccionario[value_predicted]
        result =("Cliente pertenece a {} ".format(crop))
    else:
        result =("Error! No se puede agrupar al cliente")
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)