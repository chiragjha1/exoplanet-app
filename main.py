from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import joblib, json, os
import pandas as pd
import google.generativeai as genai

# Optional: load .env file (recommended)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

BASE_DIR = Path(__file__).parent

# Load ML model and features
model = joblib.load(BASE_DIR / "model.pkl")

with open(BASE_DIR / "features.json") as f:
    features = json.load(f)

# Debug API key
api_key = os.getenv("GEMINI_API_KEY")
print("GEMINI_API_KEY loaded:", "YES" if api_key else "NO")

genai.configure(api_key=api_key)

# Use safer model name format
gemini = genai.GenerativeModel("models/gemini-1.5-flash")

# Optional: list available models (uncomment for debugging)
"""
for m in genai.list_models():
    print(m.name)
"""

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
        planet.pl_rade,
        planet.pl_bmasse,
        planet.pl_orbsmax,
        planet.pl_insol,
        planet.st_teff,
        planet.st_mass,
        planet.st_rad
    ]], columns=features)

    esi = round(float(model.predict(input_df)[0]), 4)

    # Make sure prompt is not empty
    prompt = f"""
    You are an astrophysics expert.

    A planet has the following Earth Similarity Index (ESI): {esi}.

    Based on this, generate a short insight about its habitability,
    surface conditions, and potential to support life.
    Keep it concise and engaging.
    """

    try:
        response = gemini.generate_content(prompt)

        # Safer extraction of text
        if hasattr(response, "text") and response.text:
            insight = response.text
        else:
            insight = str(response)

    except Exception as e:
        print("Gemini error:", e)
        insight = f"Insight generation failed: {str(e)}"

    tier, color = classify(esi)

    return {
        "esi": esi,
        "tier": tier,
        "color": color,
        "insight": insight
    }

def classify(esi):
    if esi >= 0.8:
        return "HIGH POTENTIAL", "#00ffcc"
    if esi >= 0.6:
        return "MODERATE POTENTIAL", "#f0c040"
    if esi >= 0.4:
        return "LOW POTENTIAL", "#ff8c42"
    return "UNLIKELY HABITABLE", "#ff4466"

@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE_DIR / "index.html").read_text()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
