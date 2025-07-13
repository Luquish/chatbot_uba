import re
from typing import Optional


def get_intent_response(query: str) -> Optional[str]:
    """Return a predefined response for simple intents.

    If the query is about the bot identity, its capabilities or is a greeting,
    return the appropriate response. Otherwise return ``None`` to allow the RAG
    pipeline to handle the query.
    """
    clean_query = query.lower().strip()

    # Consultas sobre el nombre/identidad del bot
    name_queries = [
        "cómo te llamás", "como te llamas", "¿cómo te llamas?", "¿como te llamas?",
        "cómo te llamas?", "como te llamas?", "cuál es tu nombre",
        "cual es tu nombre", "¿cuál es tu nombre?", "¿cual es tu nombre?",
        "quién eres", "quien eres", "¿quién eres?", "¿quien eres?",
        "cómo te dicen", "como te dicen", "¿cómo te dicen?", "¿como te dicen?",
        "tu nombre", "cómo te llaman", "como te llaman", "cuál es tu apellido",
        "cual es tu apellido"
    ]
    if clean_query in name_queries or any(p in clean_query for p in name_queries):
        return (
            "👨‍⚕️ Me llamo DrCecim. Soy un asistente virtual "
            "especializado en información académica de la Facultad de Medicina de la Universidad de Buenos Aires."
        )

    # Preguntas sobre las capacidades del bot
    capability_patterns = [
        r"qué.*pod[ée]s hacer", r"en qué me pod[ée]s ayudar",
        r"qué información ten[é|e]s", r"para qué serv[í|i]s",
        r"sobre qué temas", r"qué temas", r"qué materias",
        r"cuáles son tus funciones"
    ]
    for pat in capability_patterns:
        if re.search(pat, clean_query):
            return (
                "👨‍⚕️ Puedo ayudarte con consultas sobre:\n"
                "- Reglamento académico de la Facultad de Medicina 📚\n"
                "- Condiciones de regularidad para alumnos 📋\n"
                "- Régimen disciplinario y sanciones 🎓\n"
                "- Trámites administrativos para estudiantes 📄\n"
                "- Requisitos académicos y normativas 📌"
            )

    # Saludos
    greeting_words = [
        "hola", "buenas", "buen día", "buen dia", "buenos días", "buenos dias",
        "buenas tardes", "buenas noches", "saludos", "qué tal", "que tal",
        "como va", "cómo va"
    ]
    if clean_query in greeting_words or any(clean_query.startswith(g) for g in greeting_words):
        return (
            "👨‍⚕️ ¡Hola! Soy DrCecim, tu asistente de la Facultad de Medicina. "
            "¿En qué puedo ayudarte hoy?"
        )

    return None
