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
    gender = request.form['Gender']
    age = request.form['Age']
    annual_income = request.form['annualIncome']
    spending_score = request.form['spendingScore']

    feature_list = ([[gender, age, annual_income, spending_score]])
    # single_pred = np.array(feature_list).reshape(1, -1)

    transformed_features = ct.transform(feature_list)
    transformed_features[:,2:] = sc.transform(transformed_features[:,2:])
    prediction = model.predict(transformed_features).reshape(1,-1)

    # diccionario = {1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3", 4: "Cluster 4", 5: "Cluster 5"}

    # if prediction[0] in diccionario:
    #     crop = diccionario[prediction[0]]
    #     result =("Lo mejor para plantar es: {} ".format(crop))
    # else:
    #     result =("Sorry, Hoy no se come")
    return render_template('index.html',result = prediction[0])




# python main
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)