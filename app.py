from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
from pathlib import Path
from contextlib import asynccontextmanager
import warnings
warnings.filterwarnings('ignore')

# Import the tools we built in Phase 3 and Phase 4
from phase3_primary_model import train_match_engine
from phase4_tournament_simulator import extract_latest_stats

# Global variables to hold our trained model and team stats in memory
match_model = None
stats = None
base_dir = Path(__file__).resolve().parent

# This lifespan function runs once when the server starts up
@asynccontextmanager
async def lifespan(app: FastAPI):
    global match_model, stats
    print("Loading AI Model and Team Stats...")
    match_model, final_df = train_match_engine()
    stats = extract_latest_stats(final_df)
    print("AI Engine is ready!")
    yield
    print("Shutting down AI Engine...")

# Initialize FastAPI
app = FastAPI(title="FIFA ML Predictor API", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=base_dir / "static"), name="static")

# MUST INCLUDE CORS: This allows your custom UI (running somewhere else) 
# to talk to this API without the browser blocking it.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change "*" to your UI's exact URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected JSON payload from your UI
class MatchRequest(BaseModel):
    home_team: str
    away_team: str
    is_neutral: bool = False

@app.get("/")
def read_root():
    html_path = base_dir / "static" / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/teams")
def get_teams():
    if not stats:
        return JSONResponse({"teams": []})

    teams = sorted(stats.keys())
    return {"teams": teams}

@app.post("/predict")
def predict_match(request: MatchRequest):
    home = request.home_team.strip().title()
    away = request.away_team.strip().title()
    
    if home == away:
        raise HTTPException(status_code=400, detail="Home and Away teams must be different!")
        
    # Get stats (fallback to averages if the team is completely unknown)
    stat_home = stats.get(home, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    stat_away = stats.get(away, {'elo': 1500, 'form': 1.5, 'attack': 1.0})
    
    # Calculate context for the model
    features = pd.DataFrame([{
        'elo_diff': stat_home['elo'] - stat_away['elo'],
        'form_diff': stat_home['form'] - stat_away['form'],
        'attack_diff': stat_home['attack'] - stat_away['attack'],
        'is_home_advantage': 0 if request.is_neutral else 1
    }])
    
    # Get ML probabilities
    probs = match_model.predict_proba(features)[0]
    
    # Return the data as clean JSON to your frontend UI
    return {
        "home_team": home,
        "away_team": away,
        "is_neutral": request.is_neutral,
        "probabilities": {
            "home_win": round(probs[2] * 100, 1),
            "draw": round(probs[1] * 100, 1),
            "away_win": round(probs[0] * 100, 1)
        },
        "insights": {
            "elo_advantage_points": round(features['elo_diff'][0], 1),
            "form_advantage_points": round(features['form_diff'][0], 2)
        }
    }
