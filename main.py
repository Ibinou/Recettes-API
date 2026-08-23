from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
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
import gc

app = FastAPI(title="API Recettes Ultimate")

API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY) if API_KEY else None


# ============================================================
#  OUTILS RÉSEAU
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
#  EXTRACTION SCHEMA.ORG/RECIPE
# ============================================================

def extract_schema_recipe(soup: BeautifulSoup) -> Optional[dict]:
    """Retourne un dict brut {titre, ingredients, etapes, image} si trouvé, sinon None."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (TypeError, ValueError):
            continue

        candidates = data if isinstance(data, list) else [data]
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

            # L'image peut être une string, un dict ImageObject, ou une liste des deux
            image = None
            raw_image = item.get("image")
            if isinstance(raw_image, str):
                image = raw_image
            elif isinstance(raw_image, dict):
                image = raw_image.get("url")
            elif isinstance(raw_image, list) and raw_image:
                first = raw_image[0]
                image = first if isinstance(first, str) else first.get("url")

            if titre and (ingredients or etapes):
                return {
                    "titre": titre,
                    "ingredients": ingredients,
                    "etapes": etapes,
                    "image": image,
                    "prep_time": item.get("prepTime"),
                    "cook_time": item.get("cookTime"),
                    "total_time": item.get("totalTime"),
                    "portions": item.get("recipeYield"),
                }

    return None


def find_og_image(soup: BeautifulSoup) -> Optional[str]:
    for prop in ["og:image:secure_url", "og:image"]:
        tag = soup.find("meta", property=prop)
        if tag and tag.get("content"):
            return tag["content"]
    return None


# ============================================================
#  MODÈLES — schéma que Gemini doit produire (sans l'image,
#  qu'on renseigne nous-mêmes côté Python)
# ============================================================

class Ingredient(BaseModel):
    nom: str
    quantite: Optional[float] = None
    unite: Optional[str] = None
    emoji: Optional[str] = None


class RecetteIA(BaseModel):
    est_une_recette: bool
    titre: Optional[str] = None
    temps_preparation: Optional[int] = None  # en minutes
    temps_cuisson: Optional[int] = None
    temps_total: Optional[int] = None
    portions_base: Optional[int] = None
    ingredients: Optional[List[Ingredient]] = None
    etapes: Optional[List[str]] = None


class ExtractRequest(BaseModel):
    type: str
    content: str
    source: Optional[str] = None   # ex: "TikTok", "Pinterest", "Web"... affiché tel quel dans l'UI
    image: Optional[str] = None    # si le client a déjà une image sous la main (ex: poster vidéo)


PROMPT_BASE = (
    "Analyse ce contenu et réponds STRICTEMENT selon le schéma JSON demandé.\n"
    "- Si ce n'est pas une recette, mets est_une_recette à false et laisse le reste vide.\n"
    "- Sinon, extrais la recette avec précision, même si les informations sont éparpillées "
    "ou incomplètes.\n"
    "- Si c'est la photo d'un plat déjà terminé (pas de texte de recette), identifie le plat "
    "et génère une recette plausible pour le refaire.\n"
    "- temps_preparation / temps_cuisson / temps_total sont des NOMBRES DE MINUTES (entiers). "
    "Convertis les durées ISO 8601 (ex: 'PT23M') ou les mentions textuelles ('20 min') en minutes. "
    "Laisse null si vraiment introuvable.\n"
    "- portions_base est le nombre de personnes pour lequel la recette est prévue à l'origine "
    "(mets 4 par défaut si vraiment introuvable).\n"
    "- Pour chaque ingrédient : quantite est un nombre (ou null si non quantifiable, ex: 'au goût'), "
    "unite est une unité courte ('g', 'ml', 'cuillère à soupe'...) ou null pour les éléments comptables "
    "(ex: 3 œufs -> quantite: 3, unite: null), emoji est UN SEUL emoji représentatif de l'ingrédient.\n"
    "- etapes est une liste de chaînes de texte, une par étape, sans numérotation manuelle."
)


def _download_video(url: str, output_path: str) -> dict:
    """Télécharge la vidéo et retourne les métadonnées yt-dlp (dont 'thumbnail')."""
    ydl_opts = {
        "format": "worst[ext=mp4]/worst",
        "outtmpl": output_path,
        "quiet": True,
        "max_filesize": 25 * 1024 * 1024,
        "noplaylist": True,
        "http_headers": BROWSER_HEADERS,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
    return info or {}


def _upload_and_wait(path: str):
    uploaded = client.files.upload(file=path)
    while uploaded.state.name == "PROCESSING":
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    return uploaded


SOURCE_LABELS = {
    "tiktok_url": "TikTok",
    "pinterest_url": "Pinterest",
    "video_url": "Instagram",
    "image_url": "Image",
    "web_url": "Web",
    "text": "Texte",
}


@app.post("/extraire")
def extraire_recette(request: ExtractRequest):
    if not client:
        raise HTTPException(status_code=500, detail="Clé API Gemini non configurée.")

    temp_file_path = None
    uploaded_file = None
    image_finale = request.image  # priorité à l'image déjà fournie par le client

    try:
        contents = []
        base_name = f"fichier_{uuid.uuid4().hex}"

        # ------------------------------------------------------
        # 1. TIKTOK / VIDÉOS DÉJÀ IDENTIFIÉES (Instagram via WebView)
        # ------------------------------------------------------
        if request.type in ["tiktok_url", "video_url"]:
            temp_file_path = base_name + ".mp4"
            info = _download_video(request.content, temp_file_path)
            if not image_finale:
                image_finale = info.get("thumbnail")
            uploaded_file = _upload_and_wait(temp_file_path)
            contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 2. PINTEREST — vidéo en priorité, image en repli
        # ------------------------------------------------------
        elif request.type == "pinterest_url":
            try:
                temp_file_path = base_name + ".mp4"
                info = _download_video(request.content, temp_file_path)
                if not image_finale:
                    image_finale = info.get("thumbnail")
                uploaded_file = _upload_and_wait(temp_file_path)
                contents = [uploaded_file, PROMPT_BASE]
            except yt_dlp.utils.DownloadError:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

                html = fetch_html(request.content)
                soup = BeautifulSoup(html, "html.parser")
                pin_image = find_og_image(soup)

                if not pin_image:
                    return {
                        "status": "error",
                        "message": "Impossible de trouver une vidéo ou une image sur ce pin Pinterest.",
                    }

                if not image_finale:
                    image_finale = pin_image

                temp_file_path = base_name + ".jpg"
                download_file(pin_image, temp_file_path, referer=request.content)
                uploaded_file = client.files.upload(file=temp_file_path)
                contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 3. IMAGE DIRECTE
        # ------------------------------------------------------
        elif request.type == "image_url":
            if not image_finale:
                image_finale = request.content
            temp_file_path = base_name + ".jpg"
            download_file(request.content, temp_file_path)
            uploaded_file = client.files.upload(file=temp_file_path)
            contents = [uploaded_file, PROMPT_BASE]

        # ------------------------------------------------------
        # 4. TEXTE BRUT
        # ------------------------------------------------------
        elif request.type == "text":
            contents = [f"{PROMPT_BASE}\n\nCONTENU :\n{request.content[:5000]}"]

        # ------------------------------------------------------
        # 5. SITES WEB CLASSIQUES
        # ------------------------------------------------------
        elif request.type == "web_url":
            html = fetch_html(request.content)
            soup = BeautifulSoup(html, "html.parser")

            schema = extract_schema_recipe(soup)
            if schema:
                if not image_finale:
                    image_finale = schema.get("image")

                texte_structure = f"TITRE: {schema['titre']}\n"
                if schema.get("prep_time"):
                    texte_structure += f"TEMPS DE PRÉPARATION (brut) : {schema['prep_time']}\n"
                if schema.get("cook_time"):
                    texte_structure += f"TEMPS DE CUISSON (brut) : {schema['cook_time']}\n"
                if schema.get("total_time"):
                    texte_structure += f"TEMPS TOTAL (brut) : {schema['total_time']}\n"
                if schema.get("portions"):
                    texte_structure += f"PORTIONS (brut) : {schema['portions']}\n"
                texte_structure += "\nINGRÉDIENTS:\n" + "\n".join(f"- {i}" for i in schema["ingredients"])
                texte_structure += "\n\nÉTAPES:\n" + "\n".join(schema["etapes"])

                contents = [
                    f"{PROMPT_BASE}\n\n"
                    f"Voici une recette déjà structurée extraite du site (ne rien inventer, "
                    f"juste convertir/nettoyer) :\n\n{texte_structure}"
                ]
            else:
                if not image_finale:
                    image_finale = find_og_image(soup)

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

        # GÉNÉRATION — sortie JSON structurée, contrainte par le schéma RecetteIA
        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                response_mime_type="application/json",
                response_schema=RecetteIA,
            ),
        )

        parsed = json.loads(response.text)

        if not parsed.get("est_une_recette"):
            return {"status": "error", "message": "Aucune recette trouvée dans ce contenu."}

        return {
            "status": "success",
            "titre": parsed.get("titre") or "Recette sans titre",
            "source": request.source or SOURCE_LABELS.get(request.type, "Import"),
            "image": image_finale,
            "temps_preparation": parsed.get("temps_preparation"),
            "temps_cuisson": parsed.get("temps_cuisson"),
            "temps_total": parsed.get("temps_total"),
            "portions_base": parsed.get("portions_base") or 4,
            "ingredients": parsed.get("ingredients") or [],
            "etapes": parsed.get("etapes") or [],
        }

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
