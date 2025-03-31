from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import xgboost as xgb

app = Flask(__name__)

# Load pre-trained model
model_path = "personalrecommender.pkl"
with open(model_path, "rb") as file:
    best_xgb_model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        hour = int(request.form['hour'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        user_interaction_count = int(request.form['user_interaction_count'])
        item_popularity = int(request.form['item_popularity'])
        item_id = int(request.form['item_id'])
        
        # Prepare input as NumPy array
        input_data = np.array([[user_id, item_id, hour, day, month, user_interaction_count, item_popularity]])

        # Check model type
        if isinstance(best_xgb_model, xgb.Booster):
            dmatrix = xgb.DMatrix(input_data)
            score = best_xgb_model.predict(dmatrix)[0]
        else:
            score = best_xgb_model.predict(input_data)[0]
        
        return render_template('recommendations.html', recommendations=[(item_id, score)])
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
