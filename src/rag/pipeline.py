"""Etapa 3 · Arquitectura de recuperación aumentada (RAG).

Flujo de una pregunta:

    pregunta ─► embedding ─► búsqueda en la base vectorial (top-k)
             ─► ¿hay contexto suficiente? ──no──► "no encuentro esa información"
             └─sí─► prompt = instrucciones + contexto + pregunta ─► LLM ─► respuesta con fuentes

TODO etapa 3: construir_prompt, PipelineRAG e indexar_documentos.
TODO etapa 4: indexar_fragmentos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .vector_store import Resultado

PLANTILLA_PROMPT = """Eres el asistente interno de Andina Logística S.A.S.
Responde en español, de forma breve y precisa, usando ÚNICAMENTE la información del contexto.
Si la respuesta no está en el contexto, responde exactamente: "{sin_contexto}"
Al final de cada afirmación indica la fuente entre corchetes, por ejemplo [politica_vacaciones.md].

<contexto>
{contexto}
</contexto>

Pregunta: {pregunta}
Respuesta:"""

RESPUESTA_SIN_CONTEXTO = "No encuentro esa información en los documentos de la empresa."


@dataclass
class RespuestaRAG:
    pregunta: str
    texto: str
    fuentes: list[str] = field(default_factory=list)
    resultados: list[Resultado] = field(default_factory=list)
    prompt: str = ""

def formatear_contexto(resultados: list[Resultado]) -> str:
    bloques = []
    for i, r in enumerate(resultados, start=1):
        fuente = r.metadata.get("fuente", r.id)
        bloques.append(f"[{i}] (fuente: {fuente})\n{r.texto.strip()}")
    return "\n\n".join(bloques)


def construir_prompt(pregunta: str, resultados: list[Resultado]) -> str:
    return PLANTILLA_PROMPT.format(
        sin_contexto=RESPUESTA_SIN_CONTEXTO,
        contexto=formatear_contexto(resultados),
        pregunta=pregunta.strip(),
    )

class PipelineRAG:
    def __init__(self, store, llm, k: int = 4, umbral: float = 0.25) -> None:
        self.store, self.llm, self.k, self.umbral = store, llm, k, umbral

    def recuperar(self, pregunta: str, filtro: dict | None = None) -> list[Resultado]:
        resultados = self.store.buscar(pregunta, k=self.k, filtro=filtro)
        # Guardarraíl: descartamos lo que se parece muy poco a la pregunta.
        return [r for r in resultados if r.score >= self.umbral]

    def responder(self, pregunta: str, filtro: dict | None = None) -> RespuestaRAG:
        resultados = self.recuperar(pregunta, filtro)
        if not resultados:
            # Sin contexto no llamamos al modelo: así evitamos que invente.
            return RespuestaRAG(pregunta, RESPUESTA_SIN_CONTEXTO)
        prompt = construir_prompt(pregunta, resultados)
        texto = self.llm.generar(prompt)
        fuentes = list(dict.fromkeys(r.metadata.get("fuente", r.id) for r in resultados))
        return RespuestaRAG(pregunta, texto, fuentes, resultados, prompt)

def indexar_documentos(store, documentos) -> int:
    """Estrategia ingenua de la etapa 3: un documento completo = un vector."""
    store.agregar(
        ids=[d.fuente for d in documentos],
        textos=[d.texto for d in documentos],
        metadatas=[dict(d.metadata) for d in documentos],
    )
    return len(documentos)



def indexar_fragmentos(store, documentos, tamano: int = 600, solapamiento: int = 100) -> int:
    raise NotImplementedError("Etapa 4: indexar_fragmentos")
