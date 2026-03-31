# python -m venv .venv
# python -m pip install --upgrade pip
# .venv\Scripts\Activate
# pip install flask pandas prophet plotly
# python app.py
import os
from dotenv import load_dotenv 
from flask import Flask, render_template, request, jsonify
from prophet.serialize import model_from_json

load_dotenv()
app = Flask(__name__)

PORT = int(os.getenv('FLASK_PORT', 5000))
DEBUG = os.getenv('DEBUG_MODE', 'False').lower() == 'true'

# 1. LOAD THE MODEL
with open('gym_model.json', 'r') as f:
    model = model_from_json(f.read())

# 2. GENERATE GLOBAL FORECAST (Next 7 days)
future = model.make_future_dataframe(periods=168, freq='h')
forecast = model.predict(future)

def get_recommendation(day, workout, segment):
    segment_map = {
        'Early Bird': (5, 8), 'Morning': (9, 11), 'Lunch': (12, 13),
        'Afternoon': (14, 16), 'Peak': (17, 20), 'Night': (21, 23)
    }
    
    rec_df = forecast[['ds', 'yhat']].copy()
    rec_df['day_name'] = rec_df['ds'].dt.day_name()
    rec_df['hour'] = rec_df['ds'].dt.hour
    
    if day and day != "No Preference":
        rec_df = rec_df[rec_df['day_name'] == day]
    if segment and segment in segment_map:
        start, end = segment_map[segment]
        rec_df = rec_df[(rec_df['hour'] >= start) & (rec_df['hour'] <= end)]
        
    multiplier = 1.5 if workout == 'weights' else 1.0
    rec_df['score'] = rec_df['yhat'] * multiplier
    
    if rec_df.empty: return None
    
    best = rec_df.sort_values('score').iloc[0]
    return {
        "time": best['ds'].strftime('%A, %I:%M %p'),
        "occ": round(best['yhat'], 1),
        "vibe": "Quiet" if best['yhat'] < 15 else "Moderate" if best['yhat'] < 30 else "Busy"
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    result = get_recommendation(data.get('day'), data.get('workout'), data.get('segment'))
    if result:
        return jsonify(result)
    return jsonify({"error": "No slots found"}), 404

if __name__ == '__main__':
    print("\n" + "="*50)
    print(f"🚀 GYMFLOW AI SERVER IS STARTING")
    print(f"🔗 URL: http://127.0.0.1:{PORT}")
    print(f"🛠️  Debug Mode: {DEBUG}")
    print("="*50 + "\n")
    
    # Use the variables here
    app.run(debug=DEBUG, port=PORT)