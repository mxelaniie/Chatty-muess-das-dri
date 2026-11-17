# Chätty, muess das dri?  -  Streamlit-App
# Starten im Terminal (im Projektordner):  streamlit run app.py

import base64
import csv
import json
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel


# ---------------------------------------------------------------
# 1. Grundeinstellungen (gleich wie im Notebook)
# ---------------------------------------------------------------

load_dotenv()            # liest GEMINI_API_KEY aus der Datei .env
client = genai.Client()  # findet den Schlüssel automatisch
MODELL = "gemini-3.5-flash-lite"

# Modell auf dem FHNW-Server (OPENAI_API_KEY und OPENAI_BASE_URL in der .env)
MODELL_FHNW = "zai-org/GLM-5.3-Flash"

# Lokales Modell mit Ollama (läuft auf dem eigenen Computer, Fotos bleiben dort)
client_lokal = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")   # Ollama braucht keinen echten Schlüssel
MODELL_LOKAL = "gemma3:4b"

PROMPT_V1 = """Du bist Fachperson für die amtliche Vermessung im Kanton Bern.
Schau dir das Foto an und beantworte zwei Fragen:
1. Welche Einzelobjektart ist zu sehen? Wenn keine passt, wähle kein_AV_Objekt.
   Wenn mehrere Objekte zu sehen sind, nimm das wichtigste.
2. Wird das Objekt nach der Praxis im Kanton Bern in der amtlichen Vermessung erfasst?
   Wenn das aus dem Foto allein nicht sicher zu entscheiden ist, wähle abklaeren.
Begründe kurz auf Deutsch."""

# Regeln aus regeln.csv in einen Text umwandeln (wie Variante 2 im Notebook)
regeltext = ""
with open("regeln.csv", newline="", encoding="utf-8-sig") as f:
    for regel in csv.DictReader(f, delimiter=";"):
        regeltext = regeltext + "- " + regel["objektart"] + ": " + regel["regel_bern"] + " (" + regel["fundstelle"] + ")\n"

PROMPT_V2 = PROMPT_V1 + "\n\nHalte dich beim Entscheid an diese Regeln des Kantons Bern:\n" + regeltext


# So muss Gemini antworten, nur diese Werte sind erlaubt
class Antwort(BaseModel):
    objektart: Literal["Mauer", "uebriger_Gebaeudeteil", "Unterstand", "wichtige_Treppe", "kein_AV_Objekt"]
    erfassen: Literal["ja", "nein", "abklaeren"]
    begruendung: str


# ---------------------------------------------------------------
# 2. Hilfsfunktionen
# ---------------------------------------------------------------

def gps_lesen(foto):
    """Liest die GPS-Koordinaten aus dem Foto. Gibt (breite, laenge) zurück oder None."""
    bild = Image.open(foto)
    exif = bild.getexif()
    gps = exif.get_ifd(0x8825)   # 0x8825 ist die Nummer des GPS-Bereichs in den EXIF-Daten

    if 2 not in gps or 4 not in gps:   # 2 = Breite, 4 = Länge
        return None

    # Koordinaten sind als Grad, Minuten, Sekunden gespeichert
    breite = float(gps[2][0]) + float(gps[2][1]) / 60 + float(gps[2][2]) / 3600
    laenge = float(gps[4][0]) + float(gps[4][1]) / 60 + float(gps[4][2]) / 3600

    if gps.get(1) == "S":   # Südhalbkugel
        breite = -breite
    if gps.get(3) == "W":   # westlich von Greenwich
        laenge = -laenge

    return breite, laenge


def lade(datei):
    """Liest eine CSV-Datei und gibt eine Liste von Dictionaries zurück."""
    with open(datei, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter=";"))


def auswerten(antworten, labels):
    """Berechnet die Trefferquoten in Prozent (gleich wie Notebook Kapitel 4)."""
    gesamt = 0
    objekt_richtig = 0
    erfassen_richtig = 0
    ja_gesamt = 0
    ja_richtig = 0
    nein_gesamt = 0
    nein_richtig = 0

    for label in labels:
        if label["erfassen"] == "":   # Bilder ohne Label (weg) überspringen
            continue
        if label["datei"] not in antworten:
            continue

        antwort = antworten[label["datei"]]
        gesamt = gesamt + 1

        if antwort["objektart"] == label["objektart"]:
            objekt_richtig = objekt_richtig + 1
        if antwort["erfassen"] == label["erfassen"]:
            erfassen_richtig = erfassen_richtig + 1

        if label["erfassen"] == "ja":
            ja_gesamt = ja_gesamt + 1
            if antwort["erfassen"] == "ja":
                ja_richtig = ja_richtig + 1
        if label["erfassen"] == "nein":
            nein_gesamt = nein_gesamt + 1
            if antwort["erfassen"] == "nein":
                nein_richtig = nein_richtig + 1

    ja_recall = 100 * ja_richtig / ja_gesamt
    nein_recall = 100 * nein_richtig / nein_gesamt

    return {
        "Objektart": round(100 * objekt_richtig / gesamt),
        "Erfassen": round(100 * erfassen_richtig / gesamt),
        "ja erkannt": round(ja_recall),
        "nein erkannt": round(nein_recall),
        "ausgewogen": round((ja_recall + nein_recall) / 2),
    }


# ---------------------------------------------------------------
# 3. Seite aufbauen
# ---------------------------------------------------------------

st.set_page_config(page_title="Chätty, muess das dri?", page_icon="📐")
st.title("Chätty, muess das dri?")
st.write("Foto hochladen und ein KI-Modell fragen, ob das Objekt in die amtliche Vermessung (Kanton Bern) gehört.")

tab_pruefen, tab_ergebnisse = st.tabs(["Foto prüfen", "Ergebnisse"])


# ---------------- Tab 1: Foto prüfen ----------------
with tab_pruefen:
    foto = st.file_uploader("Foto auswählen", type=["png", "jpg", "jpeg"])

    if foto is not None:
        st.image(foto.getvalue(), width=400)

        # GPS aus dem Foto (nur zur Info, entscheidet nichts)
        koordinaten = gps_lesen(foto)
        if koordinaten is None:
            st.caption("Das Foto hat keine GPS-Koordinaten.")
        else:
            breite = koordinaten[0]
            laenge = koordinaten[1]
            link = ("https://map.geo.admin.ch/?lang=de&layers=ch.kantone.cadastralwebmap-farbe&swisssearch="
                    + str(round(breite, 6)) + "," + str(round(laenge, 6)))
            st.caption("Aufnahmeort (Standort der Kamera, nicht des Objekts):")
            st.map({"lat": [breite], "lon": [laenge]}, zoom=16, size=5)
            st.markdown("[Auf map.geo.admin.ch mit AV-Plan anschauen](" + link + ")")

        # Optionales Feld: öffentlicher Grund
        grund = st.radio(
            "Liegt das Objekt auf öffentlichem Grund? (optional)",
            ["weiss nicht", "ja", "nein"],
            horizontal=True,
        )

        modell_wahl = st.radio(
            "Welches Modell?",
            [
                "Gemini (Cloud)",
                "FHNW-Server (GLM-5.3-Flash)",
                "Lokal (gemma3:4b, Foto bleibt auf dem Computer)",
            ],
        )

        prompt = PROMPT_V2
        if grund == "ja":
            prompt = prompt + "\nGeprüfte Zusatzinfo: Das Objekt liegt auf öffentlichem Grund. Du musst das nicht mehr abklären, entscheide mit dieser Angabe."
        if grund == "nein":
            prompt = prompt + "\nGeprüfte Zusatzinfo: Das Objekt liegt auf privatem Grund. Du musst das nicht mehr abklären, entscheide mit dieser Angabe."

        if st.button("Prüfen"):
            ergebnis = None   # bleibt None, wenn etwas schiefgeht
            if modell_wahl == "Gemini (Cloud)":
                with st.spinner("Gemini schaut sich das Foto an ..."):
                    antwort = client.models.generate_content(
                        model=MODELL,
                        contents=[
                            types.Part.from_bytes(data=foto.getvalue(), mime_type=foto.type),
                            prompt,
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=Antwort,
                            temperature=0,
                        ),
                    )
                ergebnis = json.loads(antwort.text)
            else:
                # FHNW-Server und Ollama haben beide die OpenAI-Schnittstelle, nur Client und Modell sind anders
                if modell_wahl == "FHNW-Server (GLM-5.3-Flash)":
                    welcher_client = OpenAI()   # findet Schlüssel und Adresse des FHNW-Servers in der .env
                    welches_modell = MODELL_FHNW
                else:
                    welcher_client = client_lokal
                    welches_modell = MODELL_LOKAL

                bild_text = base64.b64encode(foto.getvalue()).decode()   # Bild als Text für die OpenAI-Schnittstelle
                with st.spinner("Das Modell schaut sich das Foto an (lokal kann das 1 bis 2 Minuten dauern) ..."):
                    try:
                        r = welcher_client.chat.completions.create(
                            model=welches_modell,
                            messages=[{
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {"type": "image_url", "image_url": {"url": "data:" + foto.type + ";base64," + bild_text}},
                                ],
                            }],
                            response_format={
                                "type": "json_schema",
                                "json_schema": {"name": "antwort", "schema": Antwort.model_json_schema()},
                            },
                            temperature=0,
                        )
                        ergebnis = json.loads(r.choices[0].message.content)
                    except Exception:   # z. B. wenn Ollama nicht läuft
                        st.error("Keine Antwort vom Modell. Lokal: Läuft Ollama (Lama-Symbol in der Taskleiste)? "
                                 "FHNW: Stehen OPENAI_API_KEY und OPENAI_BASE_URL in der .env?")

            if ergebnis is not None:
                st.subheader("Objektart: " + ergebnis["objektart"])
                if ergebnis["erfassen"] == "ja":
                    st.success("Erfassen: ja")
                if ergebnis["erfassen"] == "nein":
                    st.error("Erfassen: nein")
                if ergebnis["erfassen"] == "abklaeren":
                    st.warning("Erfassen: abklären")
                st.write(ergebnis["begruendung"])


# ---------------- Tab 2: Ergebnisse (Dashboard) ----------------
with tab_ergebnisse:
    labels = lade("labels.csv")

    # Dateiname -> Name im Dashboard
    NAMEN = {
        "antworten_v1.csv": "V1 Gemini, nur Foto",
        "antworten_v2.csv": "V2 Gemini + Regeln",
        "antworten_fhnw.csv": "FHNW GLM-5.3-Flash + Regeln",
        "antworten_lokal.csv": "Lokal gemma3:4b + Regeln",
    }

    # Alle vorhandenen Antwort-Dateien laden (fehlende werden übersprungen)
    varianten = {}
    for datei in NAMEN:
        try:
            antworten = {}
            for zeile in lade(datei):
                antworten[zeile["datei"]] = zeile
            varianten[NAMEN[datei]] = antworten
        except FileNotFoundError:
            pass

    # Trefferquoten berechnen und als Tabelle für das Diagramm aufbereiten
    zeilen = []
    tabelle = {}
    for variante in varianten:
        quoten = auswerten(varianten[variante], labels)
        tabelle[variante] = quoten
        for kennzahl in quoten:
            zeilen.append({"Kennzahl": kennzahl, "Variante": variante, "Prozent": quoten[kennzahl]})

    st.subheader("Trefferquoten in Prozent")
    st.bar_chart(zeilen, x="Kennzahl", y="Prozent", color="Variante", stack=False)
    st.caption("ausgewogen = Mittelwert aus «ja erkannt» und «nein erkannt». "
               "Bilder ohne Label (weg) zählen nicht.")
    st.table(tabelle)   # gleiche Zahlen als Tabelle (Spalten = Varianten)

    # Falsche Bilder anzeigen
    st.subheader("Wo lag das Modell falsch?")
    gewaehlt = st.selectbox("Variante", list(varianten.keys()))
    antworten = varianten[gewaehlt]

    anzahl = 0
    for label in labels:
        if label["erfassen"] == "" or label["datei"] not in antworten:
            continue
        antwort = antworten[label["datei"]]
        if antwort["objektart"] == label["objektart"] and antwort["erfassen"] == label["erfassen"]:
            continue   # alles richtig, nicht anzeigen

        anzahl = anzahl + 1
        links, rechts = st.columns([1, 2])
        with links:
            st.image("./data/bilder/" + label["datei"])
        with rechts:
            st.markdown("**" + label["datei"] + "**")
            st.write("Richtig: " + label["objektart"] + ", " + label["erfassen"])
            st.write("Modell: " + antwort["objektart"] + ", " + antwort["erfassen"])
            st.caption(antwort["begruendung"])
        st.divider()

    st.write(str(anzahl) + " Bilder mit mindestens einem Fehler.")
