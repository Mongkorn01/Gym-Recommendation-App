# 🏋️ GymFlow AI: Smart Schedule Architect

GymFlow AI is a predictive workout planning application that doesn't just tell you when the gym is busy—it builds your entire weekly routine around the quietest possible windows. By leveraging **Meta Prophet** for time-series forecasting, the app calculates the exact duration of your custom workout and finds the optimal "low-crowd" slot to fit it in.

<p align="center">
  <img src="https://github.com/user-attachments/assets/9c8ba1bf-aea5-44e8-ad5d-d63c18a7f6ca"
       alt="GymFlow AI Interface Dashboard" 
       style="max-width: 900px; width: 100%; height: auto; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
</p>

## ✨ Key Features

- **Personalized Weekly Split**: Select your training days and assign specific muscle groups (Chest, Back, Legs, etc.) to each.
- **Dynamic Duration Calculation**: Pick exactly which exercises you plan to do; the AI calculates your total session time to ensure the "quiet window" lasts for your entire workout.
- **Prophet-Powered Optimization**: Scans historical density data to suggest the absolute quietest start time within your preferred segment (Early Bird, Morning, Night, or **No Preference**).
- **Clean Schedule Export**: Generate a finalized, clutter-free training plan and download it as a `.txt` file for easy reference on your phone.
- **Real-Time Occupancy Data**: View predicted occupancy percentages and "vibe" ratings (Quiet, Moderate, Busy) for every suggested slot.

## 🚀 The 3-Step Workflow

1.  **Configure**: Choose your training days and pick your workout routine from the sidebar.
2.  **Filter**: Tick the specific exercises you want to perform. The AI dynamically sums up the estimated time required for the session.
3.  **Optimize**: The system queries the Prophet model and returns a precise Start/End time, removing all checkboxes for a clean, professional summary.

## 📸 Preview
<p align="center">
  <img src="https://github.com/user-attachments/assets/e149166a-7f4c-48fe-9ee8-302c8ae67cde" 
       alt="GymFlow AI Interface Dashboard - 2" 
       style="max-width: 700px; width: 100%; height: auto; border-radius: 8px;">
</p>
<p align="center">
  <img src="https://github.com/user-attachments/assets/6ca3a518-850f-4f67-a791-3f57226f600f"
       alt="GymFlow AI Interface Dashboard - 3" 
       style="max-width: 700px; width: 100%; height: auto; border-radius: 8px;">
</p>

## 🛠️ Local Setup & Installation

### 1. Clone the repository
```bash
git clone [https://github.com/Mongkorn01/Gym-Recommendation-App.git](https://github.com/Mongkorn01/Gym-Recommendation-App.git)
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

app.py: The Flask backend containing the Prophet forecasting logic and recommendation engine.
gym_model.json: The serialized Meta Prophet model used for generating 7-day forecasts.
templates/index.html: The modern frontend UI built with Tailwind CSS and LocalStorage persistence.
.env: Configuration file for Flask ports and environment modes.


## 🧪 How it Works

The system utilizes the Meta Prophet library to fit non-linear trends with daily and weekly seasonality. When a user requests a schedule, the backend filters the forecast by the user's preferred "Time Segment" and applies a 1.5x occupancy multiplier for weight-based workouts to ensure extra space is available for equipment-heavy sessions.

## Live Demo
👉 https://gym-recommendation-app-ryu5.onrender.com/
> ⚠️ **Note:** The app may take a few minutes to start when first opened.
---

Developed with 💙 for quiet squat racks.
