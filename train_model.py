# Version 6: Evaluation metric

# ===============================
# People Count Time Series Forecast
# Predict next 7 days (168 hours)
# ===============================

import pandas as pd
import matplotlib.pyplot as plt
import os
from prophet import Prophet
import json
from prophet.serialize import model_to_json

# ==========================================================
# 1. LOAD CSV
# ==========================================================

file_path = "gym_data.csv"

try:
    if not os.path.exists(file_path):
        raise FileNotFoundError("CSV file not found.")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("CSV file is empty.")

    print("✅ File loaded successfully!")

except Exception as e:
    print("❌ Error:", e)
    exit()

# ==========================================
# 2. Optimized Count (The Turnstile Method)
# ==========================================

# Convert to datetime first
df["checkin_time"] = pd.to_datetime(df["checkin_time"])
df["checkout_time"] = pd.to_datetime(df["checkout_time"])

entries = pd.DataFrame({"timestamp": df["checkin_time"], "change": 1})
exits = pd.DataFrame({"timestamp": df["checkout_time"], "change": -1})

# Combine, sort, and calculate running total
events = pd.concat([entries, exits]).sort_values("timestamp")
events["occupancy"] = events["change"].cumsum()

# Resample to hourly frequency
# 'asfreq' ensures we have a row for every single hour
hourly_df = events.set_index("timestamp").resample("H").last().ffill()
hourly_df.rename(columns={"occupancy": "people_count"}, inplace=True)

# Drop the 'change' column if it exists in the final df
hourly_df = hourly_df[["people_count"]].fillna(0)

# -------------------------------
# 3. Ensure hourly frequency
# -------------------------------

# Prophet needs columns named 'ds' and 'y'
prophet_df = hourly_df.reset_index()
prophet_df.columns = ['ds', 'y']
# ==========================================
# 4. Train Prophet Model
# ==========================================
print("Training Prophet model...")

# We enable daily and weekly seasonality specifically for gym traffic
model = Prophet(
    changepoint_prior_scale=0.05,
    yearly_seasonality=False,
    weekly_seasonality=True,
    daily_seasonality=True
)

model.fit(prophet_df)
print("Model training complete")

# ==========================================
# 5. Forecast next 7 days (168 hours)
# ==========================================
future = model.make_future_dataframe(periods=168, freq='H')
forecast = model.predict(future)

# ==========================================
# 6. Plot Results (Customized)
# ==========================================

# 6.1 Main Forecast Plot
only_forecast = forecast.tail(168)

fig1 = model.plot(only_forecast)
ax = fig1.gca()  # Get current axes

# Set Titles and Labels
ax.set_title("Gym Occupancy Forecast: Next 7 Days", fontsize=16, fontweight='bold')
ax.set_xlabel("Date and Time", fontsize=12)
ax.set_ylabel("Number of People", fontsize=12)

# Optional: Adjust X-axis limits to ensure it looks tight
ax.set_xlim([only_forecast['ds'].min(), only_forecast['ds'].max()])

plt.show()

# 6.2 Components Plot (Trends, Weekly, Daily)
fig2 = model.plot_components(forecast)

# The components plot has multiple subplots, so we iterate through them
titles = ["Overall Trend", "Weekly Pattern", "Daily Pattern"]
for i, ax in enumerate(fig2.get_axes()):
    ax.set_title(titles[i], fontsize=14, loc='left', color='blue')
    ax.set_ylabel("Occupancy Adjustment")

plt.tight_layout()
plt.show()

# ==========================================
# 7. Extract the Forecast for Use
# ==========================================
# 'yhat' is the predicted value
forecast_result = forecast[['ds', 'yhat']].tail(168)
print("\nNext 7 days forecast:")
print(forecast_result)

# ==========================================
# 8. Recommendation system
# ==========================================
def get_smart_recommendation(forecast_df, day=None, segment=None, pref_time=None, workout_type=None):
    """
    All parameters are optional.
    Segments: 'Early Bird', 'Morning', 'Lunch', 'Afternoon', 'Peak', 'Night'
    """
    # 1. Prepare Data
    rec_df = forecast_df[['ds', 'yhat']].copy()
    rec_df['day_name'] = rec_df['ds'].dt.day_name()
    rec_df['hour'] = rec_df['ds'].dt.hour

    # 2. Define Segment Windows
    segment_map = {
        'Early Bird': (5, 8),
        'Morning': (9, 11),
        'Lunch': (12, 13),
        'Afternoon': (14, 16),
        'Peak': (17, 20),
        'Night': (21, 23)
    }

    # 3. Apply Optional Filters
    if day:
        rec_df = rec_df[rec_df['day_name'].str.lower() == day.lower()]

    if segment and segment in segment_map:
        start, end = segment_map[segment]
        rec_df = rec_df[(rec_df['hour'] >= start) & (rec_df['hour'] <= end)]

    if pref_time is not None:
        # Narrow down to a 3-hour window around their preferred hour
        rec_df = rec_df[(rec_df['hour'] >= pref_time - 1) & (rec_df['hour'] <= pref_time + 1)]

    # 4. Apply Workout Type 'Sensitivity'
    # We don't filter the data, we just adjust the "Crowd Score"
    # Weights users are more sensitive to high occupancy (yhat)
    multiplier = 1.5 if workout_type == 'weights' else 1.0
    rec_df['score'] = rec_df['yhat'] * multiplier

    # 5. Get the Best Result
    if rec_df.empty:
        return "No slots match those specific filters. Try loosening your constraints!"

    best_slot = rec_df.sort_values('score').iloc[0]

    return {
        "Recommended Time": best_slot['ds'].strftime('%A, %I:%M %p'),
        "Predicted Occupancy": round(best_slot['yhat'], 1),
        "Vibe": "Quiet" if best_slot['yhat'] < 15 else "Moderate" if best_slot['yhat'] < 30 else "Busy"
    }

from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

# ==========================================
# 9. Evaluation (Backtesting)
# ==========================================

# 'initial': How much data to use for the first training (e.g., 30 days)
# 'period': How often to perform a new forecast (e.g., every 7 days)
# 'horizon': How far into the "future" to predict for each fold (7 days)
df_cv = cross_validation(
    model, 
    initial='30 days', 
    period='7 days', 
    horizon='7 days',
    parallel="processes" # Speeds up calculation
)

# Calculate standard metrics (MAE, RMSE, MAPE, etc.)
df_p = performance_metrics(df_cv)
print("\n--- Standard Metrics ---")# Calculate standard metrics (MAE, RMSE, etc.)
df_p = performance_metrics(df_cv)

print("\n--- Available Performance Metrics ---")
# This automatically prints all columns that were successfully calculated
print(df_p.head()) 

# If you want to be specific but safe:
cols_to_show = [c for c in ['horizon', 'mae', 'rmse', 'mape'] if c in df_p.columns]
print("\n--- Summary Table ---")
print(df_p[cols_to_show].head())

# ==========================================
# 10. Calculating MASE (Custom)
# ==========================================

def calculate_mase(training_data, cv_df):
    # 1. Calculate MAE of the forecast
    mae = np.mean(np.abs(cv_df['y'] - cv_df['yhat']))
    
    # 2. Calculate MAE of the "Naive" forecast on training data
    # (The average absolute difference between consecutive actual values)
    naive_mae = np.mean(np.abs(np.diff(training_data['y'])))
    
    return mae / naive_mae

mase_score = calculate_mase(prophet_df, df_cv)
print(f"\n✅ MASE Score: {round(mase_score, 4)}")

if mase_score < 1:
    print("Interpretation: Your model is performing better than a naive baseline!")
else:
    print("Interpretation: The model is struggling; a naive 'last-hour' forecast is more accurate.")

# ==========================================
# 10. Visualizing
# ==========================================
# Visualizing MAPE over the 7-day horizon
from prophet.plot import plot_cross_validation_metric

# Change 'mape' to 'mae' since 'mape' doesn't exist in your results
try:
    fig3 = plot_cross_validation_metric(df_cv, metric='mae')
    plt.title("Mean Absolute Error (MAE) over 7-Day Horizon")
    plt.show()
except Exception as e:
    print(f"Could not plot metric: {e}")

# ==========================================
# TEST CASES
# ==========================================

# Case A: Very specific
print(get_smart_recommendation(forecast, day='Monday', segment='Peak', workout_type='weights'))

# Case B: Just wants the best time for weights any day
print(get_smart_recommendation(forecast, workout_type='weights'))

# Case C: Wants to go at 6 PM (18:00) but doesn't care which day
print(get_smart_recommendation(forecast, pref_time=18))

# ==========================================
# Export model
# ==========================================

# 1. Define where you want to save it (Current folder is default)
filename = 'gym_model.json'

# 2. Serialize and write the file directly to your disk
try:
    with open(filename, 'w') as f:
        f.write(model_to_json(model))
    
    # Get the full path so you know exactly where it landed
    full_path = os.path.abspath(filename)
    print(f"✅ Model saved successfully!")
    print(f"📍 Location: {full_path}")

except Exception as e:
    print(f"❌ Failed to save model: {e}")
