#!/usr/bin/env python3
"""
Análisis detallado de problemas con Google Sheets en el catálogo.
"""

import yaml
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, List

def analyze_sheets_catalog():
    """Analiza el catálogo de sheets y genera reporte de problemas."""
    
    # FIX: Cargar .env desde el directorio raíz del proyecto
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    load_dotenv(env_path)
    
    print("🔍 ANÁLISIS DE GOOGLE SHEETS CATALOG")
    print("=" * 60)
    print(f"📁 Cargando .env desde: {env_path}")
    print(f"✅ CURSOS_SPREADSHEET_ID cargado: {bool(os.getenv('CURSOS_SPREADSHEET_ID'))}")
    print()
    
    # Cargar catálogo
    with open("/Users/lucamazzarello_/Desktop/Repositories/chatbot_uba/config/sheets_catalog.yaml", 'r') as f:
        catalog = yaml.safe_load(f)
    
    domains = catalog.get('domains', {})
    
    print(f"📊 RESUMEN GENERAL:")
    print(f"   - Total dominios configurados: {len(domains)}")
    print(f"   - Archivo catálogo: ✅ Válido")
    print()
    
    # Análisis por dominio
    problems = {
        "permission_errors": [],
        "range_format_errors": [],
        "missing_env_vars": [],
        "configuration_issues": []
    }
    
    print("📋 ANÁLISIS POR DOMINIO:")
    print("-" * 40)
    
    for domain_name, config in domains.items():
        print(f"\n🔸 {domain_name.upper()}")
        
        # Verificar spreadsheet_id
        spreadsheet_id = config.get('spreadsheet_id', '')
        if spreadsheet_id.startswith('ENV:'):
            env_var = spreadsheet_id.split(':', 1)[1]
            resolved_id = os.getenv(env_var)
            if resolved_id:
                print(f"   📄 ID: {resolved_id} (desde {env_var})")
            else:
                print(f"   ❌ Variable de entorno {env_var} no definida")
                problems["missing_env_vars"].append(f"{domain_name}: {env_var}")
        else:
            print(f"   📄 ID: {spreadsheet_id}")
        
        # Verificar configuración de rango
        sheet_name = config.get('sheet_name', 'Hoja 1')
        ranges = config.get('ranges', {})
        default_range = ranges.get('default', 'A:E')
        print(f"   📊 Hoja: '{sheet_name}' | Rango: {default_range}")
        
        # Verificar triggers
        triggers = config.get('triggers', {})
        keywords = triggers.get('keywords', [])
        print(f"   🔍 Keywords: {len(keywords)} configuradas")
        
        # Determinar problema basado en los logs del test anterior
        if domain_name == 'cursos':
            problems["range_format_errors"].append(f"{domain_name}: Problema con formato de rango 'Hoja 1'!A:E")
        else:
            problems["permission_errors"].append(f"{domain_name}: Sin permisos de acceso (403)")
    
    print("\n" + "=" * 60)
    print("🚨 PROBLEMAS IDENTIFICADOS:")
    print("=" * 60)
    
    # Errores de permisos
    if problems["permission_errors"]:
        print(f"\n❌ ERRORES DE PERMISOS (403 - Permission Denied):")
        for error in problems["permission_errors"]:
            print(f"   - {error}")
        print("\n   💡 SOLUCIONES:")
        print("   1. Hacer las sheets PÚBLICAS (Anyone with the link can view)")
        print("   2. O compartir específicamente con la cuenta de servicio")
        print("   3. Verificar que los IDs de las sheets son correctos")
    
    # Errores de formato de rango
    if problems["range_format_errors"]:
        print(f"\n❌ ERRORES DE FORMATO DE RANGO:")
        for error in problems["range_format_errors"]:
            print(f"   - {error}")
        print("\n   💡 SOLUCIONES:")
        print("   1. Verificar que el nombre de la hoja es correcto")
        print("   2. Usar formato sin comillas: Hoja1!A:E en lugar de 'Hoja 1'!A:E")
        print("   3. O cambiar el nombre de la hoja para que no tenga espacios")
    
    # Variables de entorno faltantes  
    if problems["missing_env_vars"]:
        print(f"\n❌ VARIABLES DE ENTORNO FALTANTES:")
        for error in problems["missing_env_vars"]:
            print(f"   - {error}")
        print("\n   💡 SOLUCIÓN:")
        print("   Definir las variables en el archivo .env")
    
    # Recomendaciones generales
    print("\n" + "=" * 60)
    print("💡 RECOMENDACIONES PARA SOLUCIONARLO:")
    print("=" * 60)
    
    print("\n1️⃣ PERMISOS DE SHEETS:")
    print("   - Abrir cada Google Sheet")
    print("   - Hacer clic en 'Share' > 'Change to anyone with the link'")
    print("   - Cambiar permisos a 'Viewer'")
    
    print("\n2️⃣ NOMBRES DE HOJAS:")
    print("   - Cambiar nombres de hojas para evitar espacios")
    print("   - Ejemplo: 'Hoja 1' → 'Hoja1' o 'Sheet1'")
    print("   - O actualizar el catálogo con nombres correctos")
    
    print("\n3️⃣ VERIFICAR IDS:")
    print("   - Confirmar que todos los IDs de spreadsheet son correctos")
    print("   - Abrir cada URL manualmente para verificar acceso")
    
    print("\n4️⃣ VARIABLES DE ENTORNO:")
    print("   - Revisar archivo .env para variables faltantes")
    print("   - Ejemplo: CURSOS_SPREADSHEET_ID=1LbsmdSYS9UFaWtSwObYJQey-mCnFnlaT-rPzdwMbwbg")
    
    # URLs para verificar manualmente
    print("\n" + "=" * 60)
    print("🔗 URLs PARA VERIFICAR MANUALMENTE:")
    print("=" * 60)
    
    for domain_name, config in domains.items():
        spreadsheet_id = config.get('spreadsheet_id', '')
        if spreadsheet_id.startswith('ENV:'):
            env_var = spreadsheet_id.split(':', 1)[1]
            spreadsheet_id = os.getenv(env_var, f"[{env_var}_NOT_SET]")
        
        if spreadsheet_id and not spreadsheet_id.startswith('['):
            url = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit"
            print(f"   • {domain_name}: {url}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_sheets_catalog()