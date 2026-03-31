# 🏋️ GymFlow AI: Occupancy Forecasting

GymFlow AI is a predictive analytics web application that helps users find the perfect time to hit the gym. By leveraging **Meta Prophet** for time-series forecasting, the app predicts crowd density and suggests the quietest slots based on specific workout types and user preferences.

<p align="center">
  <img src="https://github.com/user-attachments/assets/53516d5f-21bd-48ba-9833-cda4a59e03b2" 
       alt="GymFlow AI Interface Dashboard" 
       style="max-width: 900px; width: 100%; height: auto; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
</p>

## ✨ Features

- **AI-Powered Predictions**: Uses historical check-in data to forecast gym occupancy for the next 7 days.
- **Smart Filtering**: Filter by preferred day and time segments (Early Bird, Peak, Night, etc.).
- **Workout Sensitivity**: Adjusts recommendations for "Free Weights" users who require more space/available racks.
- **Interactive UI**: A clean, modern dashboard built with Tailwind CSS.

## 📸 Preview
<p align="center">
  <img src="https://github.com/user-attachments/assets/dd2ea2b8-e975-423e-b34f-7b6399e3943c" 
       alt="Occupancy Forecast Analytics Chart" 
       style="max-width: 700px; width: 100%; height: auto; border-radius: 8px;">
</p>

## 🛠️ Local Setup & Installation

Follow these steps to get the environment ready and run the server on your machine.

### 1. Clone the repository
```bash
git clone https://github.com/Mongkorn01/Gym-Recommendation-App.git
cd Gym-Recommendation-App
```

### 2. Set up the Virtual Environment
```bash
# Create the environment
python -m venv .venv

# Activate it (Windows)
.venv\Scripts\activate

# Activate it (Mac/Linux)
# source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
python app.py
```
Once started, the server will be available at: http://127.0.0.1:5000 (by default)

## 📂 Project Structure
app.py: The Flask backend containing the Prophet forecasting and recommendation logic.

gym_model.json: The serialized/exported Prophet model.

templates/index.html: The frontend UI built with HTML and Tailwind CSS.

gym_data.csv: (Optional) The raw check-in/check-out data used for training.

## 🧪 How it Works
The backend utilizes the Prophet library to fit non-linear trends with daily and weekly seasonality. It effectively solves the "Connection Error" issues common in notebook environments by establishing a stable local Flask server to handle user requests.

---

Developed with 💙 for quiet squat racks.
