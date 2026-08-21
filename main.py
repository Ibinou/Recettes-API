from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
import os
import time
import uuid
from google import genai
from google.genai import types

app = FastAPI(title="API Recettes")

# La clé sera cachée dans les variables d'environnement de Render
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

class VideoRequest(BaseModel):
    url: str

@app.post("/extraire")
def extraire_recette(request: VideoRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée sur le serveur.")
        
    temp_video = f"vid_{uuid.uuid4().hex}.mp4"
    video_file = None
    
    try:
        ydl_opts = {'format': 'worst', 'outtmpl': temp_video, 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(request.url, download=True)
            desc = info.get('description', '')
            titre = info.get('title', 'Recette')

        video_file = client.files.upload(file=temp_video)
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = client.files.get(name=video_file.name)

        prompt = f'Analyse la vidéo et ce texte : "{desc[:500]}". Si pas une recette, réponds "PAS_UNE_RECETTE". Sinon, format : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...'
        response = client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(temperature=0.2, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        
        res = response.text.strip()
        if res == "PAS_UNE_RECETTE":
            return {"status": "error", "message": "Pas une recette."}
            
        return {"status": "success", "titre": titre, "recette": res}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_video): os.remove(temp_video)
        try: client.files.delete(name=video_file.name)
        except: pass
