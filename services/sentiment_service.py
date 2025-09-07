"""
Servicio de análisis de sentimiento básico para el chatbot UBA.
Implementa análisis simple sin dependencias externas costosas.
"""
import re
import logging
from typing import Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Resultado del análisis de sentimiento."""
    sentiment: str  # 'positive', 'negative', 'neutral', 'urgent'
    confidence: float
    emotional_indicators: list
    suggested_tone: str


class SentimentService:
    """Servicio de análisis de sentimiento básico para consultas universitarias."""
    
    def __init__(self):
        # Diccionarios de palabras para análisis de sentimiento
        self.positive_words = {
            'gracias', 'perfecto', 'excelente', 'genial', 'bueno', 'buena', 'buenos', 'buenas',
            'bien', 'ok', 'okey', 'dale', 'listo', 'entendido', 'claro', 'perfecto',
            'joya', 'bárbaro', 'buenísimo', 'fantástico', 'maravilloso', 'increíble'
        }
        
        self.negative_words = {
            'problema', 'problemas', 'error', 'errores', 'mal', 'mala', 'malos', 'malas',
            'terrible', 'horrible', 'pésimo', 'desastre', 'confuso', 'confusa', 'difícil',
            'complicado', 'complicada', 'frustrante', 'molesto', 'molesta', 'enojado',
            'enojada', 'furioso', 'furiosa', 'indignado', 'indignada', 'preocupado',
            'preocupada', 'ansioso', 'ansiosa', 'estresado', 'estresada'
        }
        
        self.urgent_words = {
            'urgente', 'urgentemente', 'rápido', 'rápidamente', 'ya', 'ahora', 'inmediatamente',
            'emergencia', 'emergencias', 'crítico', 'crítica', 'importante', 'importantes',
            'necesito', 'necesitamos', 'ayuda', 'ayúdame', 'socorro', 'desesperado',
            'desesperada', 'desesperación', 'pánico', 'pánico', 'asustado', 'asustada'
        }
        
        self.polite_words = {
            'por favor', 'disculpe', 'disculpa', 'perdón', 'perdona', 'gracias',
            'muchas gracias', 'mil gracias', 'agradezco', 'agradecido', 'agradecida'
        }
        
        # Patrones para detectar emociones específicas
        self.emotion_patterns = {
            'frustration': [
                r'no entiendo',
                r'no me sale',
                r'estoy confundido',
                r'estoy confundida',
                r'no sé qué hacer',
                r'no encuentro',
                r'no funciona'
            ],
            'urgency': [
                r'necesito.*ya',
                r'urgente.*ayuda',
                r'para.*ayer',
                r'lo antes posible',
                r'sin demora'
            ],
            'confusion': [
                r'no entiendo',
                r'no está claro',
                r'confuso',
                r'confusa',
                r'no sé',
                r'no tengo idea'
            ],
            'gratitude': [
                r'muchas gracias',
                r'mil gracias',
                r'te agradezco',
                r'me ayudaste',
                r'perfecto',
                r'excelente'
            ]
        }
    
    def analyze_sentiment(self, query: str) -> SentimentResult:
        """
        Analiza el sentimiento de una consulta del usuario.
        
        Args:
            query: La consulta del usuario
            
        Returns:
            SentimentResult: Resultado del análisis
        """
        query_lower = query.lower().strip()
        
        # Detectar palabras clave
        positive_count = sum(1 for word in self.positive_words if word in query_lower)
        negative_count = sum(1 for word in self.negative_words if word in query_lower)
        urgent_count = sum(1 for word in self.urgent_words if word in query_lower)
        polite_count = sum(1 for word in self.polite_words if word in query_lower)
        
        # Detectar patrones emocionales
        emotional_indicators = self._detect_emotional_patterns(query_lower)
        
        # Determinar sentimiento principal
        sentiment, confidence = self._determine_sentiment(
            positive_count, negative_count, urgent_count, polite_count, emotional_indicators
        )
        
        # Sugerir tono de respuesta
        suggested_tone = self._suggest_response_tone(sentiment, emotional_indicators)
        
        return SentimentResult(
            sentiment=sentiment,
            confidence=confidence,
            emotional_indicators=emotional_indicators,
            suggested_tone=suggested_tone
        )
    
    def _detect_emotional_patterns(self, query_lower: str) -> list:
        """Detecta patrones emocionales específicos en la consulta."""
        detected_emotions = []
        
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    detected_emotions.append(emotion)
                    break
        
        return detected_emotions
    
    def _determine_sentiment(self, positive_count: int, negative_count: int, 
                           urgent_count: int, polite_count: int, 
                           emotional_indicators: list) -> Tuple[str, float]:
        """Determina el sentimiento principal y su confianza."""
        
        # Si hay urgencia, priorizar
        if urgent_count > 0 or 'urgency' in emotional_indicators:
            return 'urgent', min(0.9, 0.6 + (urgent_count * 0.1))
        
        # Si hay frustración o confusión, es negativo
        if 'frustration' in emotional_indicators or 'confusion' in emotional_indicators:
            return 'negative', 0.8
        
        # Si hay gratitud, es positivo
        if 'gratitude' in emotional_indicators:
            return 'positive', 0.8
        
        # Comparar conteos
        total_emotional_words = positive_count + negative_count
        
        if total_emotional_words == 0:
            return 'neutral', 0.5
        
        if positive_count > negative_count:
            confidence = min(0.9, 0.5 + (positive_count * 0.1))
            return 'positive', confidence
        elif negative_count > positive_count:
            confidence = min(0.9, 0.5 + (negative_count * 0.1))
            return 'negative', confidence
        else:
            return 'neutral', 0.6
    
    def _suggest_response_tone(self, sentiment: str, emotional_indicators: list) -> str:
        """Sugiere el tono apropiado para la respuesta."""
        
        if sentiment == 'urgent':
            return 'empathetic_urgent'
        elif sentiment == 'negative':
            if 'frustration' in emotional_indicators:
                return 'patient_explanatory'
            elif 'confusion' in emotional_indicators:
                return 'clear_supportive'
            else:
                return 'empathetic_helpful'
        elif sentiment == 'positive':
            return 'friendly_encouraging'
        else:
            return 'professional_neutral'
    
    def adjust_response_for_sentiment(self, response: str, sentiment_result: SentimentResult) -> str:
        """
        Ajusta una respuesta basándose en el análisis de sentimiento.
        
        Args:
            response: Respuesta original
            sentiment_result: Resultado del análisis de sentimiento
            
        Returns:
            str: Respuesta ajustada
        """
        if sentiment_result.sentiment == 'urgent':
            # Agregar indicación de urgencia
            response = f"🚨 {response}"
            if "no tengo información" not in response.lower():
                response += "\n\nSi necesitas ayuda inmediata, contacta a @cecim.nemed por Instagram."
        
        elif sentiment_result.sentiment == 'negative':
            if 'frustration' in sentiment_result.emotional_indicators:
                response = f"Entiendo tu frustración. {response}"
            elif 'confusion' in sentiment_result.emotional_indicators:
                response = f"Te explico paso a paso: {response}"
            else:
                response = f"Lamento que tengas este problema. {response}"
        
        elif sentiment_result.sentiment == 'positive':
            # Mantener tono positivo
            if not any(word in response.lower() for word in ['gracias', 'perfecto', 'excelente']):
                response += "\n\n¡Espero que te sea útil!"
        
        return response


# Instancia global del servicio
sentiment_service = SentimentService()
