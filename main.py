from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import yt_dlp
import os
import time
import uuid
import json
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from google import genai
from google.genai import types
import gc  # Le module de nettoyage de la mémoire

app = FastAPI(title="API Recettes Ultimate")

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# ============================================================
#  OUTILS RÉSEAU — session réutilisable avec retries + headers
#  qui imitent un vrai navigateur (évite pas mal de blocages)
# ============================================================

def build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.mount("http://", HTTPAdapter(max_retries=retries))
    return session


HTTP_SESSION = build_session()

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
}


def fetch_html(url: str, timeout: int = 15) -> str:
    """Récupère le HTML d'une page en imitant un vrai navigateur, avec retries automatiques."""
    response = HTTP_SESSION.get(url, headers=BROWSER_HEADERS, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response.text


def download_file(url: str, path: str, referer: Optional[str] = None):
    headers = dict(BROWSER_HEADERS)
    if referer:
        headers["Referer"] = referer
    rep = HTTP_SESSION.get(url, headers=headers, stream=True, timeout=20)
    rep.raise_for_status()
    with open(path, "wb") as f:
        for chunk in rep.iter_content(chunk_size=8192):
            f.write(chunk)


# ============================================================
#  EXTRACTION SCHEMA.ORG/RECIPE — la vraie amélioration robustesse
#  Beaucoup de sites (Marmiton, 750g, blogs pro) exposent déjà
#  leurs recettes dans ce format structuré, pour le SEO.
#  C'est plus fiable, plus léger, et moins cher en tokens qu'un
#  scraping brut de toute la page.
# ============================================================

def extract_schema_recipe(soup: BeautifulSoup) -> Optional[str]:
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (TypeError, ValueError):
            continue

        candidates = data if isinstance(data, list) else [data]

        # Certains sites imbriquent le Recipe dans un @graph
        flattened = []
        for item in candidates:
            if isinstance(item, dict) and "@graph" in item:
                flattened.extend(item["@graph"])
            else:
                flattened.append(item)

        for item in flattened:
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type", "")
            types_list = item_type if isinstance(item_type, list) else [item_type]
            if "Recipe" not in types_list:
                continue

            titre = item.get("name", "")
            ingredients = item.get("recipeIngredient", []) or []

            instructions_raw = item.get("recipeInstructions", []) or []
            etapes = []
            for step in instructions_raw:
                if isinstance(step, dict):
                    texte_etape = step.get("text", "")
                    if texte_etape:
                        etapes.append(texte_etape)
                elif isinstance(step, str):
                    etapes.append(step)

            if titre and (ingredients or etapes):
                texte = f"TITRE: {titre}\n\nINGRÉDIENTS:\n"
                texte += "\n".join(f"- {i}" for i in ingredients)
                texte += "\n\nÉTAPES:\n"
                texte += "\n".join(f"{idx + 1}. {e}" for idx, e in enumerate(etapes))
                return texte

    return None


def find_og_image(soup: BeautifulSoup) -> Optional[str]:
    """Cherche l'image principale d'une page (utile pour Pinterest notamment)."""
    for prop in ["og:image:secure_url", "og:image"]:
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            return tag["content"]
    return None


class ExtractRequest(BaseModel):
    type: str
    content: str


PROMPT_BASE = (
    'Analyse ce contenu. Si ce n\'est pas une recette, réponds "PAS_UNE_RECETTE". '
    'Sinon, extrais la recette avec précision, même si les informations sont éparpillées '
    'ou incomplètes. Si c\'est la photo d\'un plat déjà terminé (pas de texte de recette), '
    'identifie le plat et génère une recette plausible pour le refaire. '
    'Format attendu : TITRE: ... INGRÉDIENTS: ... ÉTAPES: ...'
)


@app.post("/extraire")
def extraire_recette(request: ExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")

    temp_file_path = None
    uploaded_file = None

    try:
        contents = []
        base_name = f"fichier_{uuid.uuid4().hex}"

        # ------------------------------------------------------
        # 1. TIKTOK / VIDÉOS DÉJÀ IDENTIFIÉES (ex: Instagram via WebView)
        # ------------------------------------------------------
        if request.type in ["tiktok_url", "video_url"]:
            temp_file_path = base_name + ".mp4"
            _download_video(request.content, temp_file_path)
            uploaded_file = _upload_and_wait(temp_file_path)
            contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 2. PINTEREST — géré entièrement côté serveur maintenant.
        #    On tente d'abord la vidéo (yt-dlp la supporte nativement),
        #    et si c'est un pin photo (pas de vidéo), on se rabat
        #    automatiquement sur l'image principale de la page.
        #    Plus besoin de passer par la WebView, donc plus fiable.
        # ------------------------------------------------------
        elif request.type == "pinterest_url":
            try:
                temp_file_path = base_name + ".mp4"
                _download_video(request.content, temp_file_path)
                uploaded_file = _upload_and_wait(temp_file_path)
                contents = [uploaded_file, PROMPT_BASE]
            except yt_dlp.utils.DownloadError:
                # Pas de vidéo trouvée -> c'est probablement un pin photo
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                html = fetch_html(request.content)
                soup = BeautifulSoup(html, "html.parser")
                image_url = find_og_image(soup)

                if not image_url:
                    return {
                        "status": "error",
                        "message": "Impossible de trouver une vidéo ou une image sur ce pin Pinterest.",
                    }

                temp_file_path = base_name + ".jpg"
                download_file(image_url, temp_file_path, referer=request.content)
                uploaded_file = client.files.upload(file=temp_file_path)
                contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 3. IMAGE DIRECTE (ex: photo prise par l'utilisateur, ou
        #    image déjà extraite côté client pour une autre plateforme)
        # ------------------------------------------------------
        elif request.type == "image_url":
            temp_file_path = base_name + ".jpg"
            download_file(request.content, temp_file_path)
            uploaded_file = client.files.upload(file=temp_file_path)
            contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 4. TEXTE BRUT (collé par l'utilisateur, ou légende Instagram
        #    récupérée en repli si aucune vidéo n'a été trouvée)
        # ------------------------------------------------------
        elif request.type == "text":
            contents = [f"{PROMPT_BASE}\n\nCONTENU :\n{request.content[:5000]}"]

        # ------------------------------------------------------
        # 5. SITES WEB CLASSIQUES (Marmiton, blogs, etc.)
        #    On cherche D'ABORD le JSON-LD schema.org/Recipe
        #    (fiable, structuré, peu de tokens) avant de se rabattre
        #    sur un scraping texte brut plus fragile.
        # ------------------------------------------------------
        elif request.type == "web_url":
            html = fetch_html(request.content)
            soup = BeautifulSoup(html, "html.parser")

            recette_structuree = extract_schema_recipe(soup)
            if recette_structuree:
                # On a des données propres et fiables : pas besoin de l'IA
                # pour "deviner" la structure, juste pour vérifier/nettoyer.
                contents = [
                    f"Voici une recette déjà structurée extraite du site. "
                    f"Vérifie/nettoie le format sans inventer d'informations :\n\n{recette_structuree}"
                ]
            else:
                # Repli : scraping texte brut classique
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.extract()
                texte = soup.get_text(separator=" ", strip=True)
                if len(texte) < 200:
                    return {
                        "status": "error",
                        "message": "Ce site ne renvoie pas assez de contenu lisible (page protégée ou générée en JavaScript).",
                    }
                contents = [f"{PROMPT_BASE}\n\nCONTENU DU SITE :\n{texte[:30000]}"]

        else:
            raise HTTPException(status_code=400, detail="Format non supporté.")

        if not contents:
            return {
                "status": "error",
                "message": "Impossible d'extraire un contenu exploitable depuis ce lien.",
            }

        # GÉNÉRATION
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )

        res = response.text.strip()
        if res == "PAS_UNE_RECETTE":
            return {"status": "error", "message": "Aucune recette trouvée dans ce contenu."}

        titre = res.split("\n")[0].replace("TITRE:", "").strip()
        return {"status": "success", "titre": titre, "recette": res}

    except yt_dlp.utils.DownloadError:
        return {
            "status": "error",
            "message": "Impossible de récupérer cette vidéo (lien privé, supprimé, ou non supporté).",
        }
    except requests.exceptions.RequestException:
        return {
            "status": "error",
            "message": "Le site est injoignable ou a mis trop de temps à répondre.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if uploaded_file:
            try:
                client.files.delete(name=uploaded_file.name)
            except Exception:
                pass
        gc.collect()


# ============================================================
#  HELPERS VIDÉO
# ============================================================

def _download_video(url: str, output_path: str):
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": output_path,
        "quiet": True,
        "max_filesize": 25 * 1024 * 1024,
        "noplaylist": True,
        # Ces deux options aident yt-dlp à mieux passer certains blocages
        "http_headers": BROWSER_HEADERS,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.extract_info(url, download=True)


def _upload_and_wait(path: str):
    uploaded = client.files.upload(file=path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded
