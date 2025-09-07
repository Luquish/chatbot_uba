import logging
from typing import Any, Dict, List, Optional, Tuple
from services.metrics_service import metrics_service

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - fallback cuando PyYAML no está instalado
    yaml = None

from .tools.base import BaseTool, Decision, ToolResult


logger = logging.getLogger(__name__)


class Router:
    """
    Router configurable para seleccionar herramientas (Calendar/FAQ/RAG/Sheets, etc.)
    de manera declarativa. Si no puede cargar configuración YAML o no está disponible
    PyYAML, funciona con configuración mínima por defecto.
    """

    def __init__(self, tools: List[BaseTool], config_path: Optional[str] = None):
        self.tools = tools
        self.config: Dict[str, Any] = self._load_config(config_path)

        # Propagar configuración específica a cada herramienta
        tool_configs = {}
        if self.config and 'tools' in self.config:
            for entry in self.config['tools']:
                name = entry.get('name')
                if name:
                    tool_configs[name] = entry

        for tool in self.tools:
            tool_config = tool_configs.get(tool.name, {}) if tool_configs else {}
            tool.configure(tool_config)

    def _load_config(self, config_path: Optional[str]) -> Optional[Dict[str, Any]]:
        if not config_path:
            # Intento por defecto
            config_path = 'config/router.yaml'
        try:
            if yaml is None:
                logger.warning("PyYAML no disponible; Router funcionará con configuración por defecto mínima")
                return None
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                logger.info("Configuración de Router cargada desde %s", config_path)
                return config
        except FileNotFoundError:
            logger.warning("Archivo de configuración de Router no encontrado en %s", config_path)
            return None
        except Exception as e:
            logger.warning("No se pudo cargar configuración de Router (%s): %s", config_path, str(e))
            return None

    def plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Evalúa todas las herramientas y construye un plan ordenado por score.
        Devuelve un dict con 'steps': [{tool, score, params, reasons}].
        """
        context = context or {}
        decisions: List[Tuple[BaseTool, Decision]] = []
        for tool in self.tools:
            try:
                decision = tool.can_handle(query, context)
                decisions.append((tool, decision))
            except Exception as e:
                logger.error("Error evaluando herramienta %s: %s", tool.name, str(e))

        # Ordenar por score descendente y prioridad (si la herramienta la define)
        steps = []
        for tool, decision in sorted(
            decisions,
            key=lambda x: (x[1].score, getattr(x[0], 'priority', 0)),
            reverse=True,
        ):
            steps.append({
                'tool': tool,
                'tool_name': tool.name,
                'score': decision.score,
                'params': decision.params,
                'reasons': decision.reasons,
            })

        return {'steps': steps}

    def execute_plan(self, query: str, context: Optional[Dict[str, Any]], plan: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Ejecuta el mejor paso del plan si supera umbrales internos de la herramienta.
        Si falla o no hay resultados, intenta siguientes pasos como fallback.
        Devuelve un dict estándar listo para exponer por RAGSystem.
        """
        context = context or {}
        steps: List[Dict[str, Any]] = plan.get('steps', [])
        metrics_service.record_query()
        for step in steps:
            tool: BaseTool = step['tool']
            score: float = step['score']
            params: Dict[str, Any] = step['params'] or {}

            try:
                if not tool.accepts(score):
                    continue
                result: ToolResult = tool.execute(query, params, context)
                if result and result.response:
                    logger.info(f"Router ejecutó tool={tool.name} score={score:.2f}")
                    metrics_service.inc_tool(tool.name)
                    return {
                        'query': query,
                        'response': result.response,
                        'sources': result.sources or [],
                        'relevant_chunks': result.metadata.get('relevant_chunks', []) if result.metadata else [],
                        'query_type': f"router_{tool.name}",
                        'tool_metadata': result.metadata or {},
                    }
            except Exception as e:
                logger.error("Fallo ejecutando herramienta %s: %s", tool.name, str(e))

        logger.info("Router fallback: ningún tool produjo respuesta")
        metrics_service.inc_fallback()
        return None

    def route(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Método principal de routing que combina plan + ejecución.
        
        Args:
            query: La consulta del usuario
            context: Contexto adicional (opcional)
            
        Returns:
            Dict con respuesta y metadatos, o None si no se pudo procesar
        """
        try:
            # Crear plan de ejecución
            plan = self.plan(query, context)
            
            # Ejecutar el plan
            result = self.execute_plan(query, context, plan)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en Router.route(): {str(e)}")
            metrics_service.inc_fallback()
            return None


