from flask import Flask,render_template,request,redirect,url_for
import pickle
import numpy as np

app=Flask(__name__)

#model the trained model and encoder
model= pickle.load(open('finalModel.pkl','rb'))
encoder= pickle.load(open('finalEncoder.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/result',methods=['POST'])
def result():
    if request.method=='POST':
        sex=request.form['sex']
        bp=request.form['bp']
        cholesterol=request.form['cholesterol']
        age_binned=request.form['age_binned']
        na_to_k_binned=request.form['na_to_k_binned']

        user_input=np.array([[sex,bp,cholesterol,age_binned,na_to_k_binned]])
        transformed_input=encoder.transform(user_input)
        prediction=model.predict(transformed_input)

        return render_template('result.html',prediction=prediction[0])

if __name__=='__main__':
    app.run(debug=True)


