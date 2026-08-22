from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yt_dlp
import os
import time
import uuid
import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import gc # Le module de nettoyage de la mémoire

app = FastAPI(title="API Recettes Ultimate")

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

class ExtractRequest(BaseModel):
    type: str 
    content: str

@app.post("/extraire")
def extraire_recette(request: ExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
        
    temp_file = f"fichier_{uuid.uuid4().hex}"
    uploaded_file = None
    
    try:
        contents = []
        prompt = 'Analyse ce contenu. Si ce n\'est pas une recette, réponds "PAS_UNE_RECETTE". Sinon, extrais la recette avec précision. Format attendu : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...'

        # 1. TIKTOK / VIDÉOS BRUTES (Optimisation extrême de la RAM)
        if request.type in ["tiktok_url", "video_url"]:
            temp_file += ".mp4"
            ydl_opts = {
                'format': 'worst[ext=mp4]/worst', # La pire qualité possible
                'outtmpl': temp_file,
                'quiet': True,
                'max_filesize': 25 * 1024 * 1024, # Coupe le téléchargement si > 25 Mo
                'noplaylist': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(request.content, download=True)
            
            uploaded_file = client.files.upload(file=temp_file)
            while uploaded_file.state.name == "PROCESSING":
                time.sleep(2)
                uploaded_file = client.files.get(name=uploaded_file.name)
            contents = [uploaded_file, prompt]

        # 2. PINTEREST IMAGE 
        elif request.type == "image_url":
            temp_file += ".jpg"
            rep = requests.get(request.content, stream=True)
            with open(temp_file, 'wb') as f:
                for chunk in rep.iter_content(chunk_size=8192): # Téléchargement par petits morceaux
                    f.write(chunk)
            uploaded_file = client.files.upload(file=temp_file)
            contents = [uploaded_file, prompt]

        # 3. SITES WEB CLASSIQUES (Marmiton, Blogs)
        elif request.type == "web_url":
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            rep = requests.get(request.content, headers=headers, timeout=15)
            soup = BeautifulSoup(rep.text, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.extract()
            texte = soup.get_text(separator=' ', strip=True)
            contents = [f"{prompt}\n\nCONTENU DU SITE :\n{texte[:30000]}"]

        else:
            raise HTTPException(status_code=400, detail="Format non supporté.")

        # GÉNÉRATION
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite", 
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        
        res = response.text.strip()
        if res == "PAS_UNE_RECETTE":
            return {"status": "error", "message": "Aucune recette trouvée dans ce contenu."}
            
        titre = res.split('\n')[0].replace("TITRE:", "").strip()
        return {"status": "success", "titre": titre, "recette": res}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # NETTOYAGE AGRESSIF DE LA MÉMOIRE
        if os.path.exists(temp_file): 
            os.remove(temp_file)
        if uploaded_file:
            try: client.files.delete(name=uploaded_file.name)
            except: pass
            
        # On force Python à vider la RAM immédiatement
        gc.collect()
