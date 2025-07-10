import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Load data for Bryce Harper
PLAYER_ID = 547180
SEASON = 2025
image_url = image_url = (
    f"https://img.mlbstatic.com/mlb-photos/image/upload/"
    f"w_168,d_people:generic:headshot:silo:current.png,"
    f"q_auto:best,f_auto/v1/people/{PLAYER_ID}/headshot/67/current"
)



@st.cache_data
def get_game_log(player_id, season):
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}/stats"
    params = {
        "stats": "gameLog",
        "group": "hitting",
        "season": season,
        "gameType": "R"
    }
    r = requests.get(url, params=params).json()
    splits = r['stats'][0]['splits']
    df = pd.json_normalize(splits)
    df['date'] = pd.to_datetime(df['date'])
    df['hits'] = df['stat.hits'].astype(int)
    df['at_bats'] = df['stat.atBats'].astype(int)
    return df[['date', 'hits', 'at_bats']].sort_values('date').reset_index(drop=True)

# Feature engineering
def prepare_features(df):
    df['hit_binary'] = (df['hits'] >= 1).astype(int)
    df['hits_prev'] = df['hits'].shift(1)
    df['hits_3avg'] = df['hits'].rolling(3).mean().shift(1)
    df['hits_5avg'] = df['hits'].rolling(5).mean().shift(1)
    df['days_since_last'] = df['date'].diff().dt.days.fillna(0)
    df = df.dropna().reset_index(drop=True)
    return df

# Train model
def train_model(df):
    X = df[['hits_prev', 'hits_3avg', 'hits_5avg', 'days_since_last']]
    y = df['hit_binary']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Predict next game hit
def predict_next(df, model):
    latest = df.iloc[-1]
    new_data = pd.DataFrame([{
        'hits_prev': latest['hits'],
        'hits_3avg': df['hits'].tail(3).mean(),
        'hits_5avg': df['hits'].tail(5).mean(),
        'days_since_last': 1  # Assume 1 day
    }])
    return model.predict(new_data)[0]

# Streamlit app
st.title("🔮 Phillies Player Prediction")
st.subheader("Will Bryce Harper get a hit in his next game?")

df = get_game_log(PLAYER_ID, SEASON)
df = prepare_features(df)
model = train_model(df)
proba = model.predict_proba(pd.DataFrame([{
    'hits_prev': df.iloc[-1]['hits'],
    'hits_3avg': df['hits'].tail(3).mean(),
    'hits_5avg': df['hits'].tail(5).mean(),
    'days_since_last': 1
}]))[0][1]

st.metric("Chance of Getting a Hit", f"{proba * 100:.1f}%")
st.image(image_url, caption="Bryce Harper", width=150)
st.line_chart(df.set_index('date')['hits'])

st.caption("Powered by MLB Stats API and Streamlit")