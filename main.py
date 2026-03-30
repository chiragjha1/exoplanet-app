from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import joblib, json, os
import pandas as pd
import google.generativeai as genai

BASE_DIR = Path(__file__).parent

model = joblib.load(BASE_DIR / "model.pkl")

with open(BASE_DIR / "features.json") as f:
    features = json.load(f)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini = genai.GenerativeModel("gemini-1.5-flash")

app = FastAPI()

class PlanetInput(BaseModel):
    pl_rade: float
    pl_bmasse: float
    pl_orbsmax: float
    pl_insol: float
    st_teff: float
    st_mass: float
    st_rad: float

@app.post("/predict")
def predict(planet: PlanetInput):
    input_df = pd.DataFrame([[
        planet.pl_rade, planet.pl_bmasse, planet.pl_orbsmax,
        planet.pl_insol, planet.st_teff, planet.st_mass, planet.st_rad
    ]], columns=features)

    esi = round(float(model.predict(input_df)[0]), 4)

    prompt = f"""..."""

    try:
        insight = gemini.generate_content(prompt).text
    except:
        insight = "Insight generation failed."

    tier, color = classify(esi)

    return {"esi": esi, "tier": tier, "color": color, "insight": insight}

def classify(esi):
    if esi >= 0.8: return "HIGH POTENTIAL", "#00ffcc"
    if esi >= 0.6: return "MODERATE POTENTIAL", "#f0c040"
    if esi >= 0.4: return "LOW POTENTIAL", "#ff8c42"
    return "UNLIKELY HABITABLE", "#ff4466"

@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)