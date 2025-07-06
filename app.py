#!/usr/bin/env python3
# ==============================================================
# AudioCovidApp –  servidor local embebido + auto-browser
# Autor: <Brajan Nieto Espinoza> • 05-jul-2025
# ==============================================================

import os, sys, tempfile, json, threading, webbrowser
from collections import Counter
from typing import Tuple

import numpy as np
import pandas as pd
import librosa
from flask import Flask, request, jsonify, render_template
from joblib import load
from werkzeug.utils import secure_filename
import sklearn
import sklearn.pipeline
# --------------------------------------------------------------
# 1)  Rutas de trabajo (soporta PyInstaller)
# --------------------------------------------------------------
BASE_DIR = getattr(sys, "_MEIPASS", os.path.abspath(os.path.dirname(__file__)))
APP_DIR  = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(__file__)

STATIC_FOLDER    = os.path.join(BASE_DIR, "static")
TEMPLATE_FOLDER  = os.path.join(BASE_DIR, "templates")
MODEL_PATH       = os.path.join(APP_DIR,  "svm_model.pkl")   # ← el usuario puede reemplazar este .pkl

ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm"}

# --------------------------------------------------------------
# 2)  Cargar modelo una sola vez
# --------------------------------------------------------------
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"❌ No se encontró el modelo: {MODEL_PATH}")
model = load(MODEL_PATH)

# --------------------------------------------------------------
# 3)  Utilidades de audio   (idénticas a la versión anterior)
# --------------------------------------------------------------
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    return audio / (np.max(np.abs(audio)) + 1e-6) * 0.99

def remove_silence(audio: np.ndarray, sr: int, top_db: int = 60) -> np.ndarray:
    ints = librosa.effects.split(audio, top_db=top_db)
    return np.concatenate([audio[s:e] for s, e in ints]) if ints.size else audio

def segment_audio(audio: np.ndarray, sr: int, dur=1.5, hop=0.75) -> list[np.ndarray]:
    fl, hl = int(dur * sr), int(hop * sr)
    return [audio[i:i + fl] for i in range(0, len(audio) - fl + 1, hl)]

def extract_features(seg: np.ndarray, sr: int) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=128, n_fft=512, hop_length=256)
    zcr   = librosa.feature.zero_crossing_rate(seg)
    return np.hstack([mfccs.mean(axis=1), zcr.mean()])

def process_audio_file(path: str, sex: int) -> pd.DataFrame:
    audio, sr = librosa.load(path, sr=16_000)
    audio = remove_silence(normalize_audio(audio), sr)
    if len(audio) < sr * 1.5:
        raise RuntimeError("Audio muy corto tras quitar silencios")

    feats = [extract_features(seg, sr) for seg in segment_audio(audio, sr)]
    df = pd.DataFrame(feats, columns=[f"mfcc_{i}" for i in range(128)] + ["zcr"])
    df.insert(0, "sex", sex)
    df.insert(0, "filename", os.path.basename(path))
    return df

def majority_vote(model, df: pd.DataFrame) -> Tuple[str, Counter]:
    try:
        X = df[model.feature_names_in_]
    except AttributeError:
        X = df.drop(columns=["filename", "sex"], errors="ignore")
    preds = model.predict(X)
    counts = Counter(preds.astype(str))
    final, _ = counts.most_common(1)[0]
    return final, counts

# --------------------------------------------------------------
# 4)  Servidor Flask
# --------------------------------------------------------------
app = Flask(__name__, static_folder=STATIC_FOLDER, template_folder=TEMPLATE_FOLDER)

def ok_filename(name: str) -> bool:
    return os.path.splitext(name)[1].lower() in ALLOWED_EXT

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    """Endpoint para procesar archivos de audio"""
    print("🔍 Procesando petición de upload...")
    
    if "audiofile" not in request.files:
        print("❌ No se encontró 'audiofile' en la petición")
        return jsonify(error="No file part"), 400
    
    file = request.files["audiofile"]
    if not file.filename or not ok_filename(file.filename):
        print(f"❌ Formato no soportado: {file.filename}")
        return jsonify(error="Formato no soportado"), 400
    
    try:
        sex = int(request.form.get("sex", "0"))
        assert sex in (0, 1)
    except (ValueError, AssertionError):
        print(f"❌ Valor 'sex' inválido: {request.form.get('sex')}")
        return jsonify(error="Valor 'sex' inválido"), 400

    print(f"📁 Archivo recibido: {file.filename}")
    print(f"👤 Sexo del paciente: {sex}")
    
    # Guardado temporal seguro
    suf = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suf) as tmp:
        file.save(tmp.name)
        audio_path = tmp.name
    
    try:
        # Verificar que el archivo de audio es válido
        print("🔍 Verificando archivo de audio...")
        test_audio, test_sr = librosa.load(audio_path, sr=None)
        if len(test_audio) == 0:
            raise ValueError("El archivo de audio está vacío o corrupto")
        
        print(f"✅ Audio válido - Duración: {len(test_audio)/test_sr:.2f}s, SR: {test_sr}")
        
        # Procesar el archivo
        print("🧠 Procesando con modelo de ML...")
        df = process_audio_file(audio_path, sex)
        print(f"📊 Extraídos {len(df)} segmentos de audio")
        
        # Hacer predicción
        label, counts = majority_vote(model, df)
        counts_dict = dict(counts)
        confidence = round(counts_dict[label] / sum(counts_dict.values()) * 100, 1)
        
        print(f"🎯 Predicción: {label} (confianza: {confidence}%)")
        print(f"📈 Detalles: {counts_dict}")
        
        # Preparar respuesta
        response = {
            "final_label": label,
            "confidence_pct": str(confidence),
            "segments_processed": len(df),
            "prediction_details": counts_dict,
            "audio_duration": f"{len(test_audio)/test_sr:.2f}s"
        }
        
        print("✅ Procesamiento completado exitosamente")
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Error procesando audio: {str(e)}")
        return jsonify(error=f"Error procesando audio: {str(e)}"), 500
    
    finally:
        # Limpiar archivo temporal
        try:
            os.remove(audio_path)
            print("🗑️ Archivo temporal eliminado")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar archivo temporal: {e}")

@app.route("/health")
def health():
    """Endpoint para verificar estado del servidor"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_PATH,
        "version": "1.0.0"
    })

# --------------------------------------------------------------
# 5)  Punto de entrada  (abre navegador y lanza servidor)
# --------------------------------------------------------------
def run_server():
    # waitress≈gunicorn pero cross-platform (importante en .exe)
    try:
        from waitress import serve
        print("🚀 Iniciando servidor con Waitress...")
        serve(app, host="127.0.0.1", port=5000)
    except ImportError:
        print("⚠️ Waitress no disponible, usando Flask dev server...")
        app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 AudioCovidApp - Detector de COVID-19 por Voz")
    print("=" * 60)
    print(f"📂 Directorio base: {BASE_DIR}")
    print(f"📂 Directorio app: {APP_DIR}")
    print(f"🤖 Modelo: {MODEL_PATH}")
    print(f"📁 Templates: {TEMPLATE_FOLDER}")
    print(f"📁 Static: {STATIC_FOLDER}")
    print("=" * 60)
    
    # Verificar que los directorios existen
    for name, path in [("Templates", TEMPLATE_FOLDER), ("Static", STATIC_FOLDER)]:
        if not os.path.exists(path):
            print(f"⚠️ {name} no encontrado: {path}")
    
    print("🌐 Abriendo navegador en 1 segundo...")
    # Abre el navegador 1 s después (para dar tiempo a que arranque waitress)
    threading.Timer(1.0, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    run_server()