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

app = FastAPI(title="API Recettes Multi-Sources")

# Récupération de la clé API depuis Render
API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None

class VideoRequest(BaseModel):
    url: str

def est_reseau_social(url: str) -> bool:
    """Vérifie si le lien vient d'une plateforme vidéo supportée par yt-dlp."""
    domaines = ['tiktok.com', 'instagram.com', 'facebook.com', 'fb.watch', 'pinterest', 'youtube.com']
    return any(domaine in url.lower() for domaine in domaines)

def extraire_texte_web(url: str):
    """Aspire le texte brut d'un site web classique (Marmiton, blogs, etc.)"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    rep = requests.get(url, headers=headers, timeout=15)
    soup = BeautifulSoup(rep.text, 'html.parser')
    
    titre_page = soup.title.string if soup.title else "Recette Web"
    
    # On supprime le code inutile (menus, pubs, scripts) pour économiser des tokens
    for element in soup(["script", "style", "nav", "footer", "header"]):
        element.extract()
        
    texte = soup.get_text(separator=' ', strip=True)
    return titre_page, texte[:40000] # Limite à 40 000 caractères

@app.post("/extraire")
def extraire_recette(request: VideoRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")
        
    url = request.url
    temp_video = f"vid_{uuid.uuid4().hex}.mp4"
    video_file = None
    
    try:
        titre = "Recette"
        is_video = False
        
        # --- ÉTAPE 1 : TENTATIVE VIDÉO (Réseaux Sociaux) ---
        if est_reseau_social(url):
            try:
                ydl_opts = {'format': 'worst', 'outtmpl': temp_video, 'quiet': True}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    desc = info.get('description', '')
                    titre = info.get('title', 'Recette')
                is_video = True
            except:
                # Si yt-dlp échoue (ex: c'est une simple photo sur Insta ou Pinterest), 
                # on ne plante pas, on passe au plan B.
                is_video = False

        # --- ÉTAPE 2 : PRÉPARATION DU PROMPT (Vidéo OU Texte) ---
        if is_video:
            video_file = client.files.upload(file=temp_video)
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = client.files.get(name=video_file.name)
                
            prompt = f'Analyse la vidéo et ce texte : "{desc[:500]}". Si ce n\'est pas une recette, réponds "PAS_UNE_RECETTE". Sinon, format : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...'
            contents = [video_file, prompt]
        else:
            # Plan B : Site web classique (Marmiton, Journal des Femmes, Blogs)
            titre_page, texte_page = extraire_texte_web(url)
            titre = titre_page
            prompt = f'Analyse le texte de cette page web. Si ça ne présente pas de recette de cuisine, réponds "PAS_UNE_RECETTE". Sinon, extrais-la avec ce format : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...\n\nCONTENU DU SITE :\n{texte_page}'
            contents = [prompt]
            
        # --- ÉTAPE 3 : GÉNÉRATION IA ---
        response = client.models.generate_content(
            model="gemini-3.6-flash", 
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.2, automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        )
        
        res = response.text.strip()
        
        if res == "PAS_UNE_RECETTE":
            return {"status": "error", "message": "Aucune recette n'a pu être trouvée sur ce lien."}
            
        return {"status": "success", "titre": titre, "recette": res}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # --- ÉTAPE 4 : NETTOYAGE ---
        if os.path.exists(temp_video): 
            os.remove(temp_video)
        if video_file:
            try: client.files.delete(name=video_file.name)
            except: pass
