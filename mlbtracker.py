import streamlit as st
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

# Load data for Bryce Harper
SEASON=2025
@st.cache_data
def get_teams():
    url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"  # 1 = MLB
    r = requests.get(url).json()
    teams = {team['name']: team['id'] for team in r['teams']}
    return teams

@st.cache_data
def get_team_roster(team_id):
    url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
    r = requests.get(url).json()
    
    hitters = {}
    for player in r['roster']:
        player_id = player['person']['id']
        player_name = player['person']['fullName']

        # Get player position info
        details_url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
        detail_resp = requests.get(details_url).json()
        try:
            primary_position = detail_resp['people'][0]['primaryPosition']['abbreviation']
            if primary_position != "P":  # Exclude pitchers
                hitters[player_name] = player_id
        except (KeyError, IndexError):
            continue  # Skip if missing data

    return hitters


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


def train_model(df):
    X = df[['hits_prev', 'hits_3avg', 'hits_5avg', 'days_since_last']]
    y = df['hit_binary']
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

def predict_next(df, model):
    latest = df.iloc[-1]
    new_data = pd.DataFrame([{
        'hits_prev': latest['hits'],
        'hits_3avg': df['hits'].tail(3).mean(),
        'hits_5avg': df['hits'].tail(5).mean(),
        'days_since_last': 1  # Assume 1 day
    }])
    return model.predict(new_data)[0]




teams = get_teams()
selected_team_name = st.selectbox("Select a Team", sorted(teams.keys()))
selected_team_id = teams[selected_team_name]
players = get_team_roster(selected_team_id)
selected_player_name = st.selectbox("Select a Player", sorted(players.keys()))
PLAYER_ID = players[selected_player_name]
st.subheader(f"Will {selected_player_name} get a hit in his next game?")
st.title(f"🔮 {selected_team_name} Player Prediction")
image_url = (
    f"https://img.mlbstatic.com/mlb-photos/image/upload/"
    f"w_168,d_people:generic:headshot:silo:current.png,"
    f"q_auto:best,f_auto/v1/people/{PLAYER_ID}/headshot/67/current"
)

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
st.image(image_url, caption=selected_player_name, width=150)
st.line_chart(df.set_index('date')['hits'])

st.caption("Powered by MLB Stats API and Streamlit")
