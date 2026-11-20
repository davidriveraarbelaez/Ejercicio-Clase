"""Etapa 1 · Representación vectorial de texto.

Un *embedding* convierte un texto en un vector de números de tamaño fijo.
Textos con significado parecido quedan cerca en ese espacio, y "cerca" se
mide con la similitud coseno.

TODO etapa 1: implemente las funciones y clases de este archivo siguiendo la guía.
Prueba: pytest tests/test_etapa1_embeddings.py
"""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

from .config import get_config

STOPWORDS = {
    "a", "al", "algo", "como", "con", "cual", "cuales", "cuando", "cuanto", "cuantos",
    "cuantas", "de", "del", "debo", "desde", "donde", "el", "en", "es", "esta", "este",
    "hay", "la", "las", "le", "lo", "los", "me", "mi", "mis", "no", "o", "para", "por",
    "puedo", "que", "se", "si", "sin", "sobre", "su", "sus", "tengo", "un", "una", "uno",
    "y", "ya", "yo",
}  

def normalizar(texto: str) -> str:
    """Pasa a minúsculas y quita tildes: 'Vacación' -> 'vacacion'."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


def tokenizar(texto: str) -> list[str]:
    """Palabras normalizadas, sin stopwords, recortadas a 5 letras
    (un 'stemming' mínimo: 'vacaciones' y 'vacacion' -> 'vacac')."""
    palabras = re.findall(r"[a-z0-9]+", normalizar(texto))
    return [p[:5] for p in palabras if p not in STOPWORDS and len(p) > 1]

class HashEmbedder:
    """Cada palabra cae, por un hash, en una de `dim` posiciones y suma 1 allí.
    Al final el vector se normaliza a longitud 1."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _posicion(self, palabra: str) -> int:
        digest = hashlib.md5(palabra.encode("utf-8")).hexdigest()
        return int(digest, 16) % self.dim

    def embed(self, texto: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        for palabra in tokenizar(texto):
            vector[self._posicion(palabra)] += 1.0
        norma = np.linalg.norm(vector)
        return vector / norma if norma > 0 else vector

    def embed_many(self, textos: list[str]) -> np.ndarray:
        if not textos:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed(t) for t in textos])

class SentenceTransformerEmbedder:
    """La primera vez descarga el modelo (unos 470 MB) desde Hugging Face."""

    def __init__(self, modelo: str) -> None:
        from sentence_transformers import SentenceTransformer  # importación perezosa

        self.modelo = SentenceTransformer(modelo)
        self.dim = self.modelo.get_sentence_embedding_dimension()

    def embed(self, texto: str) -> np.ndarray:
        return self.embed_many([texto])[0]

    def embed_many(self, textos: list[str]) -> np.ndarray:
        vectores = self.modelo.encode(textos, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectores, dtype=np.float32)


def get_embedder(backend: str | None = None):
    cfg = get_config()
    backend = (backend or cfg.embedding_backend).lower()
    if backend == "hash":
        return HashEmbedder()
    if backend == "local":
        return SentenceTransformerEmbedder(cfg.embedding_model)
    raise ValueError(f"EMBEDDING_BACKEND desconocido: {backend!r} (use 'hash' o 'local')")

def similitud_coseno(a: np.ndarray, b: np.ndarray) -> float:
    norma = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norma == 0:
        return 0.0
    return float(np.dot(a, b) / norma)

def mas_similares(consulta: str, textos: list[str], embedder, k: int = 3) -> list[tuple[int, float]]:
    q = embedder.embed(consulta)
    matriz = embedder.embed_many(textos)
    puntajes = [similitud_coseno(q, v) for v in matriz]
    orden = sorted(range(len(textos)), key=lambda i: puntajes[i], reverse=True)
    return [(i, puntajes[i]) for i in orden[:k]]

