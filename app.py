from flask import Flask, render_template, request
import pickle


app = Flask(__name__)
loaded_model = pickle.load(open('fixmodel.pkl', 'rb'))
text_transformer=pickle.load(open('fixtransformer.pkl', 'rb'))

def requestResults(result):
    if result == 0:
        return "Tidak Hoaks"
    else:
        return "Hoaks"
    
def sardet(text): 
    input_data = [text]
    prediction = loaded_model.predict(text_transformer.transform(input_data))
    return prediction

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    pred = sardet(message)
    result=requestResults(pred)
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False)
