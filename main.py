from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import PyPDF2
import io
import base64
from PIL import Image
import re
from typing import Optional, List

LISTA_MODELLI = [
    'gemini-flash-latest',
    'gemini-3.6-flash',
    'gemini-3.5-flash',
    'gemini-2.5-flash'
]
app = FastAPI(title="Nexus Study API", version="7.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

class RichiestaCorrezione(BaseModel):
    api_key: str
    domanda: str
    testo_risposta: Optional[str] = ""
    immagini_b64: Optional[List[str]] = []
    corso_laurea: Optional[str] = "Biotecnologie"

class RichiestaNuovaDomanda(BaseModel):
    api_key: str
    materia: str
    argomento: str

def interroga_ai_con_fallback(prompt_testo, immagini_pil=None):
    for nome_modello in LISTA_MODELLI:
        try:
            modello = genai.GenerativeModel(nome_modello)
            if immagini_pil:
                risposta = modello.generate_content([prompt_testo] + list(immagini_pil))
            else:
                risposta = modello.generate_content(prompt_testo)
            return risposta.text
        except Exception as e:
            print(f"⚠️ {nome_modello} occupato. Provo il prossimo...")
            continue
    raise Exception("Tutti i modelli AI sono momentaneamente bloccati.")

@app.get("/")
def leggi_stato_server():
    return {"status": "online"}

@app.post("/api/elabora-pdf")
async def elabora_pdf(file: UploadFile = File(...), api_key: str = Form(...), materia: str = Form(...)):
    try:
        genai.configure(api_key=api_key)
        contenuto_pdf = await file.read()
        lettore = PyPDF2.PdfReader(io.BytesIO(contenuto_pdf))
        testo_estratto = "".join([pagina.extract_text() for pagina in lettore.pages])
        
        prompt = f"""
        Agisci come professore di {materia}. Leggi attentamente l'intero documento: "{testo_estratto}"
        
        OBIETTIVO: 
        1. Scansiona l'INTERO testo e mappa TUTTI i macro-argomenti presenti. Non tralasciare i capitoli finali.
        2. Per OGNI macro-argomento, estrai SOLO le 3-5 domande FONDAMENTALI e più probabili per un esame universitario.
        
        BILANCIAMENTO ORALE/SCRITTO:
        Includi sia domande discorsive/teoriche (classiche da esame Orale, es. "Spiega il significato concettuale di..."), sia domande mirate/pratiche (da esame Scritto, es. "Scrivi l'equazione di..."). Voglio solo le domande "chiave" che ti fanno capire se lo studente merita 30 e lode.

        REGOLA SULLA COMPOSIZIONE:
        Lo studente risponderà usando una piccola lavagna digitale. Le domande devono essere dirette e circoscritte.
        È VIETATO incatenare più richieste diverse nella stessa domanda. Spezza in domande separate.

        FORMATO ESATTO RICHIESTO:
        ### ARGOMENTO: [Nome Argomento 1]
        - [Domanda chiave 1 (Orale)]
        - [Domanda chiave 2 (Scritto)]

        ### ARGOMENTO: [Nome Argomento 2]
        - [Domanda chiave 1]
        ... e così via per TUTTI gli argomenti trovati nel testo.
        """
        
        testo_risposta_ai = interroga_ai_con_fallback(prompt)
        
        righe = testo_risposta_ai.strip().split('\n')
        argomenti_estratti = {}
        argomento_corrente = "Varie"
        
        for riga in righe:
            riga = riga.strip()
            if riga.startswith("### ARGOMENTO:"):
                argomento_corrente = riga.replace("### ARGOMENTO:", "").strip()
                if argomento_corrente not in argomenti_estratti:
                    argomenti_estratti[argomento_corrente] = []
            elif riga.startswith("- "):
                domanda_testo = riga.replace("- ", "").strip()
                argomenti_estratti[argomento_corrente].append({"testo": domanda_testo, "punteggio": 0, "storico": []})
                
        return {"successo": True, "dati": argomenti_estratti}
    except Exception as e:
        return {"successo": False, "errore": str(e)}

@app.post("/api/genera-domanda")
def genera_domanda(richiesta: RichiestaNuovaDomanda):
    try:
        genai.configure(api_key=richiesta.api_key)
        prompt_nuova = f"""Sei un professore universitario. Genera UNA singola domanda d'esame sulla materia '{richiesta.materia}', focalizzata in particolare sull'argomento '{richiesta.argomento}'.
        Fai UNA delle due cose:
        1. Una domanda GENERICA/concettuale che non richieda una spiegazione lunghissima, OPPURE
        2. Una domanda PRECISA su UN SOLO procedimento o passaggio specifico.
        Restituisci SOLO il testo della domanda, senza numerazione."""
        nuova_domanda = interroga_ai_con_fallback(prompt_nuova).strip()
        return {"successo": True, "domanda": nuova_domanda}
    except Exception as e:
        return {"successo": False, "errore": str(e)}

@app.post("/api/correggi-risposta")
def correggi_risposta(richiesta: RichiestaCorrezione):
    try:
        genai.configure(api_key=richiesta.api_key)
        immagini_pil = []
        if richiesta.immagini_b64:
            for img_b64 in richiesta.immagini_b64:
                if img_b64:
                    dati_img = base64.b64decode(img_b64.split(",")[1])
                    immagini_pil.append(Image.open(io.BytesIO(dati_img)).convert("RGB"))
        
        testo_studente = richiesta.testo_risposta if richiesta.testo_risposta else "(nessuna nota testuale, vedi solo immagini)"
        contesto_corso = f'Lo studente segue il corso di laurea in "{richiesta.corso_laurea}".' if richiesta.corso_laurea else ""
        contesto_pagine = f"La risposta è su {len(immagini_pil)} lavagne/immagini separate, da leggere in ordine come un unico elaborato continuo." if len(immagini_pil) > 1 else ""

        prompt_prof = f"""Sei un tutor universitario di {richiesta.domanda}. Lo aiuti a migliorare con trucchi pratici e una valutazione costruttiva.

{contesto_corso} Calibra il taglio dei tuoi commenti sull'applicazione rilevante per quel percorso.

DOMANDA D'ESAME: "{richiesta.domanda}"
RISPOSTA DELLO STUDENTE (testo e/o immagini allegate):
Testo: "{testo_studente}"
{contesto_pagine}

ISTRUZIONI:
1. TRASCRIZIONE FEDELE: descrivi SOLO ciò che è effettivamente visibile o scritto.
2. CONFRONTO: elenca esplicitamente quali punti richiesti sono stati affrontati e quali mancano.
3. VOTO: Da 0 a 100%. Il voto deve riflettere quanto scritto rispetto a ciò che è stato EFFETTIVAMENTE richiesto.
4. COME SI SAREBBE RISPOSTO PER IL 100%.
5. CONSIGLI DA "30 E LODE" (opzionali).

FORMATO DI OUTPUT RICHIESTO:
Riga 1: SOLO il voto da 0 a 100 seguito da % (Es: 85%)
Poi:
**Cosa hai scritto:** [...]
**Cosa manca rispetto alla domanda:** [...]
**Come si sarebbe risposto per il 100%:** [...]
**Trucco Mnemonico:** [...]
**🌟 Per il 30 e lode:** [...]"""
        
        risposta_finale = interroga_ai_con_fallback(prompt_prof, immagini_pil=immagini_pil if immagini_pil else None)
        match = re.search(r'(\d{1,3})%', risposta_finale)
        voto = int(match.group(1)) if match else 0
        return {"successo": True, "voto": voto, "feedback": risposta_finale}
        
    except Exception as e:
        return {"successo": False, "errore": str(e)}
