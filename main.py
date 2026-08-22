from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
import os
import time
import uuid
import requests
from google import genai
from google.genai import types

app = FastAPI(title="API Recettes Ultimate")

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

# Nouveau modèle de données attendu par le serveur
class ExtractRequest(BaseModel):
    type: str # Peut être: "video_url", "image_url", "text", "tiktok_url"
    content: str

@app.post("/extraire")
def extraire_recette(request: ExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
        
    temp_file = f"fichier_{uuid.uuid4().hex}"
    uploaded_file = None
    
    try:
        contents = []
        prompt = 'Analyse ce contenu. Si ce n\'est pas une recette, réponds "PAS_UNE_RECETTE". Sinon, extrais la recette (si c\'est une image, lis le texte dessus). Format attendu : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...'

        # --- CAS 1 : C'EST UN LIEN VIDÉO TIKTOK (On utilise yt-dlp) ---
        if request.type == "tiktok_url" or request.type == "video_url":
            temp_file += ".mp4"
            ydl_opts = {'format': 'worst', 'outtmpl': temp_file, 'quiet': True}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(request.content, download=True)
            
            uploaded_file = client.files.upload(file=temp_file)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
            contents = [uploaded_file, prompt]

        # --- CAS 2 : C'EST UNE IMAGE PINTEREST/INSTA (Gemini Vision) ---
        elif request.type == "image_url":
            temp_file += ".jpg"
            # On télécharge l'image depuis le lien
            rep = requests.get(request.content, stream=True)
            with open(temp_file, 'wb') as f:
                f.write(rep.content)
            
            uploaded_file = client.files.upload(file=temp_file)
            contents = [uploaded_file, prompt]

        # --- CAS 3 : C'EST UN SITE WEB (Texte extrait par l'iPhone) ---
        elif request.type == "text":
            contents = [f"{prompt}\n\nCONTENU DU SITE :\n{request.content[:30000]}"]

        else:
            raise HTTPException(status_code=400, detail="Type de contenu non supporté.")

        # --- GÉNÉRATION IA ---
        response = client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        
        res = response.text.strip()
        if res == "PAS_UNE_RECETTE":
            return {"status": "error", "message": "Aucune recette trouvée dans ce contenu."}
            
        # Extraction basique du titre (la première ligne du rendu)
        titre = res.split('\n')[0].replace("TITRE:", "").strip()
        
        return {"status": "success", "titre": titre, "recette": res}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Nettoyage
        if os.path.exists(temp_file): 
            os.remove(temp_file)
        if uploaded_file:
            try: client.files.delete(name=uploaded_file.name)
            except: pass
