"""
Generador de respuestas para el sistema RAG.
Maneja la generación de respuestas contextuales usando modelos de OpenAI.
"""
import re
import random
from typing import List
from config.settings import logger
from config.constants import information_emojis
from models.openai_model import OpenAIModel
from utils.text_utils import strip_markdown_emphasis
from services.sentiment_service import sentiment_service


class ResponseGenerator:
    """Generador de respuestas inteligentes con contexto específico."""
    
    def __init__(self, primary_model, fallback_model_name, openai_api_key, 
                 api_timeout, max_output_tokens):
        self.model = primary_model
        self.fallback_model_name = fallback_model_name
        self.openai_api_key = openai_api_key
        self.api_timeout = api_timeout
        self.max_output_tokens = max_output_tokens
        
        # Clasificadores de consultas para respuestas especializadas
        self.query_classifiers = {
            'denuncias': {
                'keywords': ['denuncia', 'denunciar', 'denuncias'],
                'patterns': [
                    r'(como|cómo|de que forma|donde|dónde).*?(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                    r'(presentar?|hacer|poner|realizar|tramitar).*?(denuncia)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE DENUNCIAS:
- Menciona EXPLÍCITAMENTE que las denuncias se presentan POR ESCRITO
- Indica claramente que debe incluir una relación circunstanciada de hechos y personas
- Menciona la alternativa de presentación verbal en casos de urgencia
- Indica que la denuncia verbal debe ratificarse por escrito dentro de las 48 horas
- Si hay artículos relevantes en la información, cítalos textualmente
- Asegúrate de mencionar que la Universidad puede iniciar sumarios de oficio
"""
            },
            'regimen_disciplinario': {
                'keywords': ['régimen disciplinario', 'regimen disciplinario', 'disciplina', 'sanción', 'sancion'],
                'patterns': [
                    r'(sancion|sanción|castigo|pena|amonestacion|expulsión|suspension).*?(estudiante|alumno)',
                    r'(que|qué).*?(sancion|sanción|castigo|pena).*?(corresponde|aplica)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE RÉGIMEN DISCIPLINARIO:
- Cita con precisión los artículos relevantes del Régimen Disciplinario
- Incluye los tipos de sanciones que pueden aplicarse y su graduación
- Menciona qué autoridades pueden aplicar cada tipo de sanción
- Si se mencionan plazos o procedimientos específicos, destácalos claramente
- Explica claramente qué derechos tiene el estudiante en un proceso disciplinario
"""
            },
            'regularidad': {
                'keywords': ['regularidad', 'regular', 'condiciones'],
                'patterns': [
                    r'(como|cómo).*?(mantener|conseguir|obtener|perder).*?(regularidad|condición|condicion)',
                    r'(requisito|requisitos).*?(alumno regular|regularidad|condición)',
                ],
                'instructions': """
INSTRUCCIONES PARA CONSULTAS SOBRE REGULARIDAD:
- Destaca claramente el número mínimo de materias a aprobar en cada período
- Menciona el porcentaje máximo de aplazos permitidos
- Explica los plazos establecidos para completar la carrera
- Si hay excepciones o situaciones especiales, menciónalas
- Cita los artículos específicos sobre regularidad que sean relevantes
"""
            }
        }
    
    def generate_response(self, query: str, context: str, sources: List[str] = None) -> str:
        """
        Genera una respuesta contextualizada usando OpenAI.
        
        Args:
            query: La consulta del usuario
            context: Contexto mejorado de la consulta
            sources: Lista de fuentes consultadas
            
        Returns:
            Respuesta generada y formateada
        """
        emoji = random.choice(information_emojis)
        
        # Analizar sentimiento de la consulta
        sentiment_result = sentiment_service.analyze_sentiment(query)
        logger.info(f"Sentimiento detectado: {sentiment_result.sentiment} (confianza: {sentiment_result.confidence:.2f})")
        
        # Preparar fuentes
        sources_text = self._format_sources(sources)
        
        # Detectar tipo específico de consulta
        specific_instructions = self._get_specific_instructions(query)
        
        # Generar prompt con contexto de sentimiento
        prompt = self._build_prompt_with_sentiment(query, context, sources_text, specific_instructions, sentiment_result)
        
        # Generar respuesta con fallback
        response = self._generate_with_fallback(prompt, emoji)
        
        # Post-procesar respuesta
        response = self._post_process_response(response, query, emoji)
        
        # Ajustar respuesta según sentimiento
        response = sentiment_service.adjust_response_for_sentiment(response, sentiment_result)
        
        # Validar calidad de respuesta
        self._validate_response_quality(response, query)
        
        return response
    
    def _format_sources(self, sources: List[str]) -> str:
        """Formatea la lista de fuentes."""
        if not sources:
            return ""
        
        unique_sources = list(set(sources))
        return f"\nFUENTES CONSULTADAS:\n{', '.join(unique_sources)}"
    
    def _get_specific_instructions(self, query: str) -> str:
        """Obtiene instrucciones específicas basadas en el tipo de consulta."""
        query_lower = query.lower()
        
        for category, data in self.query_classifiers.items():
            # Verificar keywords
            if any(keyword in query_lower for keyword in data['keywords']):
                return data['instructions']
            
            # Verificar patrones regex
            for pattern in data['patterns']:
                if re.search(pattern, query_lower):
                    return data['instructions']
        
        return ""
    
    def _build_prompt(self, query: str, context: str, sources_text: str, 
                     specific_instructions: str) -> str:
        """Construye el prompt optimizado para el modelo."""
        return f"""
Eres DrCecim, asistente virtual del Centro de Estudiantes de la Facultad de Medicina UBA.

CONTEXTO ESPECÍFICO:
- Eres especialista en trámites administrativos universitarios
- Conoces profundamente el reglamento estudiantil de la UBA
- Tu audiencia son estudiantes de medicina
- Tu objetivo es resolver dudas de forma rápida y precisa

INFORMACIÓN RELEVANTE:
{context}
{sources_text}

CONSULTA: {query}
{specific_instructions}

INSTRUCCIONES ESPECÍFICAS:
1. Responde de forma concisa pero completa
2. Cita artículos específicos cuando sea relevante (ej: "Artículo 5 del reglamento...")
3. Si mencionas un procedimiento, incluye los pasos exactos
4. Para denuncias: MENCIONA EXPLÍCITAMENTE que deben ser por escrito
5. Para regularidad: Incluye números específicos (materias, porcentajes, plazos)
6. Si no tienes información completa, sugiere contactar @cecim.nemed por Instagram

FORMATO DE RESPUESTA:
- Usa viñetas (-) para listas
- Usa MAYÚSCULAS para enfatizar puntos críticos
- NO uses markdown
- Sé directo y profesional
- No hagas preguntas adicionales

IMPORTANTE: Si la información no está en los documentos proporcionados, di claramente que no tienes esa información específica.
"""
    
    def _build_prompt_with_sentiment(self, query: str, context: str, sources_text: str, 
                                   specific_instructions: str, sentiment_result) -> str:
        """Construye el prompt con contexto de sentimiento."""
        base_prompt = self._build_prompt(query, context, sources_text, specific_instructions)
        
        # Agregar contexto de sentimiento
        sentiment_context = self._get_sentiment_context(sentiment_result)
        
        return f"{base_prompt}\n\nCONTEXTO EMOCIONAL:\n{sentiment_context}"
    
    def _get_sentiment_context(self, sentiment_result) -> str:
        """Genera contexto emocional para el prompt."""
        sentiment_contexts = {
            'urgent': "El usuario parece tener una consulta URGENTE. Prioriza claridad y rapidez en la respuesta.",
            'negative': "El usuario parece frustrado o preocupado. Usa un tono empático y tranquilizador.",
            'positive': "El usuario tiene un tono positivo. Mantén un tono amigable y alentador.",
            'neutral': "El usuario tiene un tono neutral. Usa un tono profesional y directo."
        }
        
        base_context = sentiment_contexts.get(sentiment_result.sentiment, sentiment_contexts['neutral'])
        
        if sentiment_result.emotional_indicators:
            indicators_text = ", ".join(sentiment_result.emotional_indicators)
            base_context += f"\nIndicadores emocionales detectados: {indicators_text}"
        
        return base_context
    
    def _generate_with_fallback(self, prompt: str, emoji: str) -> str:
        """Genera respuesta con modelo principal y fallback si falla."""
        try:
            return self.model.generate(prompt)
        except Exception as e:
            logger.error(f"Error al generar respuesta con el modelo primario: {str(e)}")
            return self._generate_fallback_response(prompt, emoji)
    
    def _generate_fallback_response(self, prompt: str, emoji: str) -> str:
        """Genera respuesta usando modelo de fallback."""
        try:
            logger.info(f"Intentando con el modelo de respaldo: {self.fallback_model_name}")
            fallback_model = OpenAIModel(
                model_name=self.fallback_model_name,
                api_key=self.openai_api_key,
                timeout=self.api_timeout,
                max_output_tokens=self.max_output_tokens
            )
            return fallback_model.generate(prompt)
        except Exception as e2:
            logger.error(f"Error también con el modelo de respaldo: {str(e2)}")
            return f"{emoji} Lo siento, hubo un error al generar la respuesta. Por favor, intenta de nuevo o contacta a @cecim.nemed por instagram."
    
    def _post_process_response(self, response: str, query: str, emoji: str) -> str:
        """Post-procesa la respuesta generada."""
        # Limpiar formatos markdown
        response = strip_markdown_emphasis(response)
        
        # Agregar emoji si no tiene uno al inicio
        if not re.match(r'[\U0001F300-\U0001F6FF\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]', 
                       response.strip()[:1]):
            response = f"{emoji} {response}"
        
        return response
    
    def _validate_response_quality(self, response: str, query: str):
        """Valida la calidad de la respuesta para tipos específicos de consultas."""
        query_lower = query.lower()
        response_lower = response.lower()
        
        for category, data in self.query_classifiers.items():
            is_category = (
                any(keyword in query_lower for keyword in data['keywords']) or
                any(re.search(pattern, query_lower) for pattern in data['patterns'])
            )
            
            if is_category:
                self._validate_category_response(category, response_lower, response)
    
    def _validate_category_response(self, category: str, response_lower: str, response: str):
        """Valida respuesta para categoría específica."""
        validations = {
            'denuncias': {
                'check': "por escrito" not in response_lower,
                'warning': "La respuesta sobre denuncias no incluye información sobre presentación por escrito",
                'append': "\n\nRECUERDA: Las denuncias DEBEN presentarse POR ESCRITO con todos los detalles relevantes."
            },
            'regimen_disciplinario': {
                'check': not any(keyword in response_lower for keyword in ['apercibimiento', 'suspensión', 'sanción']),
                'warning': "La respuesta sobre régimen disciplinario no menciona tipos de sanciones",
                'append': None
            },
            'regularidad': {
                'check': not any(term in response_lower for term in ['materias', 'aprobar', 'porcentaje', 'plazo']),
                'warning': "La respuesta sobre regularidad no incluye información sobre requisitos clave",
                'append': None
            }
        }
        
        validation = validations.get(category)
        if validation and validation['check']:
            logger.warning(validation['warning'])
            if validation['append']:
                response += validation['append']
        
        # Validación general de calidad
        self._validate_general_quality(response_lower, response)
    
    def _validate_general_quality(self, response_lower: str, response: str):
        """Validaciones generales de calidad de respuesta."""
        # Verificar que no sea demasiado corta
        if len(response.strip()) < 20:
            logger.warning("Respuesta muy corta, puede no ser útil")
        
        # Verificar que no contenga respuestas genéricas
        generic_responses = [
            "no tengo información",
            "no puedo ayudarte",
            "no sé",
            "no entiendo"
        ]
        
        if any(generic in response_lower for generic in generic_responses):
            logger.warning("Respuesta genérica detectada, considerar mejorar el contexto")
        
        # Verificar que tenga información específica
        specific_indicators = [
            "artículo", "art.", "reglamento", "resolución", 
            "procedimiento", "pasos", "requisitos", "documentación"
        ]
        
        if not any(indicator in response_lower for indicator in specific_indicators):
            logger.info("Respuesta sin indicadores específicos, puede ser general")