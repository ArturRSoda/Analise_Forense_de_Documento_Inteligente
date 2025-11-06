#!/usr/bin/env python3
"""
Sistema Inteligente de Análise Forense de Documentos
Arquitetura: Local AI (PII Redaction) + Cloud AI (Forensic Analysis)

Fluxo simplificado:
1. Local AI identifica informações sensíveis
2. Aplicação de redação automática
3. Análise forense em nuvem
4. Relatório detalhado

Autores: Artur Luiz Rizzato, Toru Soda
UFSC - INE5448
"""

import streamlit as st
import json
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
from typing import Dict, List
import pytesseract
import base64
from ultralytics import YOLO

# Configuration
class Config:
    """Core system settings"""
    MAX_IMAGE_SIZE = 1024
    REDACTION_COLOR = (0, 0, 0)
    REDACTION_THICKNESS = -1
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# Load YOLO model
try:
    PII_MODEL = YOLO("best.pt") 
except Exception as e:
    st.error(f"Error loading local AI model: {e}")
    PII_MODEL = None

# Page configuration
st.set_page_config(
    page_title="Document Forensic Analysis",
    page_icon="🔬",
    layout="wide"
)

# === GLOBAL STYLE FOR THE ENTIRE APP ===
# === GLOBAL PROFESSIONAL DARK THEME ===
st.markdown("""
<style>
/* ----------- FONT SIZE / BASE COLORS ----------- */
html, body, [class*="css"], div, p, span, label, input, button, textarea {
    font-size: 1.05rem !important;
}
body {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    font-family: 'Segoe UI', 'Roboto', sans-serif;
}

/* ----------- HEADERS ----------- */
h1, h2, h3, h4, h5, h6 {
    color: #f8fafc !important;
    font-weight: 600 !important;
}

/* ----------- SIDEBAR ----------- */
[data-testid="stSidebar"] {
    background-color: #0a0f1a !important;
    color: #e2e8f0 !important;
}

/* ----------- BUTTONS ----------- */
.stButton>button {
    background: linear-gradient(90deg, #2563eb, #1e40af);
    color: #f8fafc !important;
    border: none;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-size: 1.05rem;
    font-weight: 500;
    box-shadow: 0 0 6px rgba(37,99,235,0.2);
    transition: all 0.2s ease-in-out;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #3b82f6, #1e3a8a);
    box-shadow: 0 0 8px rgba(59,130,246,0.3);
}

/* ----------- INPUTS ----------- */
.stTextInput>div>div>input, .stTextArea textarea {
    background-color: #1e293b !important;
    color: #f8fafc !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    font-size: 1.05rem !important;
}

/* ----------- EXPANDERS & TABS ----------- */
.streamlit-expanderHeader {
    font-size: 1.05rem !important;
    font-weight: 600 !important;
    color: #f8fafc !important;
}
[data-baseweb="tab"] {
    font-size: 1.05rem !important;
    color: #f1f5f9 !important;
}

/* ----------- IMAGES ----------- */
.stImage>img {
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.4);
}

/* ----------- JSON / CODE ----------- */
.stMarkdown pre, .stMarkdown code, .stJson {
    font-size: 0.95rem !important;
    color: #e2e8f0 !important;
    background-color: #1e293b !important;
}

/* ----------- EXPANDERS / BOXES ----------- */
.stExpander {
    background-color: #1e293b !important;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==================== CONFIGURAÇÕES ====================

class Config:
    """Configurações centralizadas"""
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    MAX_IMAGE_SIZE = 1024  # Tamanho máximo para processamento
    REDACTION_COLOR = (0, 0, 0)  # Preto para redação
    REDACTION_THICKNESS = -1  # Preenchido

# ==================== FUNÇÕES DE EXTRAÇÃO DE FEATURES ====================

def extract_visual_features(image: np.ndarray) -> Dict:
    """
    Extrai features técnicas não-sensíveis da imagem original
    Estas features NÃO contêm PII e são seguras para enviar
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Qualidade de foco (Laplacian variance)
    focus_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 2. Densidade de bordas (elementos estruturais)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # 3. Análise de cor (hologramas, marcas d'água)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_variance = float(np.std(hsv))
    
    # 4. Distribuição de brilho
    brightness_mean = float(np.mean(gray))
    brightness_std = float(np.std(gray))
    
    # 5. Análise de frequência (padrões de impressão)
    freq = np.fft.fft2(gray)
    freq_magnitude = np.abs(freq)
    high_freq_energy = float(np.sum(freq_magnitude > np.percentile(freq_magnitude, 95)))
    
    # 6. Textura (Local Binary Pattern simplificado)
    texture_score = float(np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)))
    
    # 7. Contraste
    contrast = float(gray.max() - gray.min())
    
    return {
        "quality_metrics": {
            "focus_score": round(focus_score, 2),
            "brightness_mean": round(brightness_mean, 2),
            "brightness_std": round(brightness_std, 2),
            "contrast": round(contrast, 2)
        },
        "structural_features": {
            "edge_density": round(edge_density, 4),
            "texture_complexity": round(texture_score, 2),
            "high_frequency_energy": round(high_freq_energy, 2)
        },
        "color_analysis": {
            "color_variance": round(color_variance, 2),
            "has_significant_color": bool(color_variance > 30)
        },
        "image_dimensions": {
            "height": image.shape[0],
            "width": image.shape[1]
        }
    }

# ==================== STAGE 1: LOCAL AI - PII DETECTION ====================
def detect_pii_with_yolo(image_path: Path) -> Dict:
    """
    STAGE 1 (NEW): "Direct Detection" Pipeline

    1. Loads the custom-trained YOLOv8 model.
    2. Runs the model on the full image *once*.
    3. Formats the results into the required dictionary structure.
    """
    if PII_MODEL is None:
        return {"success": False, "error": "Local PII model not loaded.", "pii_regions": []}

    try:
        # Run PII detection
        # Usamos uma confiança baixa aqui para pegar TUDO
        # A filtragem será feita pelo usuário
        results = PII_MODEL(str(image_path), conf=0.25) 

        final_pii_regions = []

        if results and len(results) > 0:
            # .xyxyn gives [x1, y1, x2, y2] normalized
            boxes_xyxyn = results[0].boxes.xyxyn.cpu().numpy()

            for i, box in enumerate(boxes_xyxyn):
                x1, y1, x2, y2 = box

                final_pii_regions.append({
                    "id": f"pii_{i}", # ID Único para o checkbox
                    "label": "pii",
                    "box": [float(y1), float(x1), float(y2), float(x2)], 
                    "source": "ai_classification"
                })

        return {
            "success": True,
            "pii_regions": final_pii_regions,
            "total_pii_found": len(final_pii_regions),
            "ai_description": f"Detected {len(final_pii_regions)} PII regions using custom YOLO model."
        }

    except Exception as e:
        st.error(f"❌ Erro fatal no Stage 1 (YOLO): {str(e)}")
        return {"success": False, "error": str(e), "pii_regions": []}


# ==================== STAGE 1: LOCAL AI - PII DETECTION (REFACTORED) ====================
# ... (Funções de fallback antigas omitidas para brevidade) ...

def get_conservative_redaction_boxes() -> List[Dict]:
    """
    Fallback: Redação baseada no layout REAL de CNH brasileira
    """
    regions = [
        {"label": "photo", "box": [0.20, 0.60, 0.55, 0.90], "source": "template_cnh"},
        {"label": "name", "box": [0.12, 0.05, 0.20, 0.55], "source": "template_cnh"},
        {"label": "cpf", "box": [0.35, 0.05, 0.42, 0.35], "source": "template_cnh"},
        {"label": "rg", "box": [0.43, 0.05, 0.50, 0.35], "source": "template_cnh"},
        {"label": "birth_date", "box": [0.52, 0.05, 0.59, 0.30], "source": "template_cnh"},
        {"label": "parents_names", "box": [0.65, 0.05, 0.80, 0.55], "source": "template_cnh"},
        {"label": "signature", "box": [0.70, 0.60, 0.88, 0.90], "source": "template_cnh"},
        {"label": "document_numbers", "box": [0.02, 0.05, 0.10, 0.55], "source": "template_cnh"}
    ]
    # Adiciona IDs únicos
    for i, region in enumerate(regions):
        region["id"] = f"template_{i}"
    return regions


def apply_redaction(image: np.ndarray, pii_regions: List[Dict]) -> np.ndarray:
    """
    Aplica redação (caixas pretas) nas regiões de PII identificadas
    """
    redacted = image.copy()
    h, w = image.shape[:2]
    
    for region in pii_regions:
        box = region['box']
        
        # Converte coordenadas normalizadas para pixels
        y1 = int(box[0] * h)
        x1 = int(box[1] * w)
        y2 = int(box[2] * h)
        x2 = int(box[3] * w)
        
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        
        if y2 <= y1 or x2 <= x1:
            continue
        
        cv2.rectangle(
            redacted,
            (x1, y1),
            (x2, y2),
            Config.REDACTION_COLOR,
            Config.REDACTION_THICKNESS
        )
    
    return redacted

def create_detection_debug_image(image: np.ndarray, pii_regions: List[Dict]) -> np.ndarray:
    """
    Cria imagem de debug mostrando as detecções SEM redação (apenas bordas)
    """
    debug = image.copy()
    h, w = image.shape[:2]
    
    for i, region in enumerate(pii_regions):
        box = region['box']
        
        y1 = int(box[0] * h)
        x1 = int(box[1] * w)
        y2 = int(box[2] * h)
        x2 = int(box[3] * w)
        
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        
        if y2 <= y1 or x2 <= x1:
            continue
        
        source = region.get('source', 'ai_classification')
        
        # Cor (sempre verde para IA, vermelho para fallback)
        color = (0, 255, 0) if source == 'ai_classification' else (0, 0, 255)
        
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
        
        # Label (número do PII)
        label = f"PII {i+1}"
        
        # Posição do texto
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
        
        # Fundo para o texto
        cv2.rectangle(debug, (x1, text_y - text_height - baseline), (x1 + text_width, text_y + baseline), color, -1)
        # Texto
        cv2.putText(debug, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    return debug

# ==================== STAGE 2: CLOUD AI - FORENSIC ANALYSIS ====================

def analyze_forensics_with_cloud(
    redacted_image_path: Path,
    original_features: Dict,
    pii_info: Dict
) -> Dict:
    """
    STAGE 2: Cloud AI analisa documento redacted para fraude
    - Prompt em Português.
    - Saída estrita em JSON seguindo o schema esperado pela UI.
    - Não gerar recomendações dinâmicas; retornar "recommendations": [].
    - Temperatura baixa e instruções para formato preciso.
    """
    try:
        import google.generativeai as genai

        if not Config.GOOGLE_API_KEY:
            return {"error": "Google API Key not configured"}

        genai.configure(api_key=Config.GOOGLE_API_KEY)

        img = Image.open(redacted_image_path)

        # Prompt fortemente guiado em PT-BR - instruções estritas de saída JSON
        prompt = f"""
Você é um perito forense especialista em documentos brasileiros (RG, CNH, CPF, Certidões).
Contexto: está analisando UMA imagem de documento com algumas áreas PII redigidas em caixas pretas. A análise deve
concentrar-se APENAS nas áreas NÃO redigidas, em aspectos de autenticidade e sinais de adulteração digital/visual.

INSTRUÇÕES IMPORTANTES:
1) RESPONDA SOMENTE EM JSON VÁLIDO (nenhum texto adicional, sem explicações fora do JSON).
2) O JSON deve obedecer exatamente o schema abaixo (campos obrigatórios). Se algo não puder ser determinado, use "Unknown" ou valores neutros.
3) NÃO gere recomendações acionáveis (a aba de recomendações será ESTÁTICA no aplicativo). Retorne "recommendations": [] sempre.
4) Use linguagem técnica forense em Português dentro dos campos de texto.
5) Foque em evidências objetivas (ex.: "bordas com artefatos JPEG próximos à caixa preta inferior direita", "microimpressão com padrão inconsistente", "rotacionamento local", "inconsistência de compressão entre regiões").
6) Seja conciso mas específico: pelo menos 4 frases na justificativa detalhada e 1 parágrafo (mínimo 5 frases) no expert_summary.

METADADOS TÉCNICOS (JSON):
{json.dumps(original_features, ensure_ascii=False)}

INFORMAÇÃO DE REDAÇÃO:
Total de Regiões Redactadas: {pii_info.get('total_pii_redacted', 'N/A')}
Total de Detecções (antes da revisão): {pii_info.get('total_pii_found', 'N/A')}

FORMATO DE SAÍDA (EXATAMENTE ESTE JSON):
{{
  "document_type": "RG|CNH|CPF|Certidao|Unknown",
  "confidence_score": 0,                     // int 0-100
  "forensic_analysis": {{
    "authenticity_verdict": "Autentico|Suspeito|Fraudulento|Unknown",
    "confidence_score": 0,                   // int 0-100
    "detailed_justification": "",            // mínimo 4 frases (Português), evidências citadas
    "background_analysis": {{
      "pattern_consistency": "consistente|inconsistente|suspeito|Unknown",
      "findings": "",
      "evidence": []
    }},
    "security_elements": {{
      "official_symbols_quality": "autentico|ruim|suspeito|Unknown",
      "detected_security_features": [],
      "missing_expected_features": [],
      "justification": ""
    }},
    "digital_forensics": {{
      "compression_artifacts": "uniforme|inconsistente|Unknown",
      "edge_analysis": "limpo|artefatos_suspeitos|Unknown",
      "cloning_patterns": "nenhum|detectado|Unknown",
      "pixel_level_anomalies": [],
      "justification": ""
    }},
    "anomalies_detected": [
      {{
        "type": "",
        "location": "",
        "severity": "alto|medio|baixo",
        "evidence": ""
      }}
    ]
  }},
  "expert_summary": "",                       // 1 parágrafo, mínimo 5 frases (Português)
  "recommendations": []                       // SEMPRE lista vazia (aplicação usa mensagem estática)
}}

Observação: responda APENAS com o JSON acima. Nada mais. Inicie a análise.
"""

        # Gera a resposta com baixa aleatoriedade
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(
            [prompt, img],
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=8192,
            )
        )

        # O SDK tende a expor .text; tratamos o texto e tentamos carregar JSON
        text = response.text if hasattr(response, "text") else str(response)
        try:
            result = json.loads(text)
        except Exception:
            # retorno de fallback: encapsula o texto bruto para debug
            return {
                "error": "Cloud model did not return valid JSON",
                "raw_response": text,
                "processing_stage": "cloud_forensic",
                "success": False
            }

        # Garantias de campos mínimos e normalização
        if 'recommendations' not in result:
            result['recommendations'] = []
        result['processing_stage'] = 'cloud_forensic'
        result['privacy_preserved'] = True

        return result

    except Exception as e:
        return {
            "error": f"Cloud forensic analysis failed: {str(e)}",
            "processing_stage": "cloud_forensic",
            "success": False
        }


# ==================== UI DISPLAY FUNCTIONS ====================

def display_hybrid_results(
    original_image: np.ndarray,
    redacted_image: np.ndarray,
    pii_info: Dict,
    features: Dict,
    forensic_result: Dict
):
    """Interface final com visual escuro, fontes maiores e exibição das 3 imagens."""

    st.markdown("### 🖼️ Comparativo das Imagens")

    st.markdown("""
    <style>
    .image-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: center;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .image-col {
        flex: 1 1 30%;
        max-width: 600px;
        text-align: center;
    }
    .image-col img {
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0,0,0,0.4);
        width: 100%;
        height: auto;
    }
    .image-caption {
        color: #f1f5f9;
        font-size: 1rem;
        margin-top: 0.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Convert to base64 safely (fix color shift)
    def img_to_b64(img):
        _, buffer = cv2.imencode('.png', img)
        return base64.b64encode(buffer).decode()

    debug_img = create_detection_debug_image(original_image, pii_info.get('all_pii_regions', []))

    orig_b64 = img_to_b64(original_image)
    detect_b64 = img_to_b64(debug_img)
    redact_b64 = img_to_b64(redacted_image)

    # Render all three side-by-side
    st.markdown(f"""
    <div class="image-row">
        <div class="image-col">
            <img src="data:image/png;base64,{orig_b64}">
            <div class="image-caption">📄 Original</div>
        </div>
        <div class="image-col">
            <img src="data:image/png;base64,{detect_b64}" style="border: 2px solid #60a5fa;">
            <div class="image-caption">🔍 Detecções de PII</div>
        </div>
        <div class="image-col">
            <img src="data:image/png;base64,{redact_b64}">
            <div class="image-caption">🕵️ Imagem Enviada (Redacted)</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Centered download button below
    _, buf = cv2.imencode('.png', redacted_image)
    st.download_button(
        "📥 Baixar Imagem Enviada (PNG)",
        buf.tobytes(),
        file_name=f"redacted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
        mime="image/png",
        use_container_width=True
    )


    with st.expander("📊 Features Técnicas Extraídas (Enviadas para Cloud)"):
        st.json(features)

    # === Erros de análise ===
    if 'error' in forensic_result:
        st.error(f"❌ Erro na análise forense: {forensic_result['error']}")
        return

    # === Tabs ===
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Veredito Forense", "🔍 Análise Detalhada", "⚠️ Anomalias", "💡 Recomendações"])

    # --- TAB 1: Veredito Forense ---
    with tab1:
        forensic = forensic_result.get('forensic_analysis', forensic_result)
        verdict = forensic.get('authenticity_verdict', 'Unknown')
        confidence = forensic.get('confidence_score', 0)

        # Determine box style by verdict
        if verdict.lower() in ['autentico', 'autêntico']:
            css_class, icon, label = "authentic", "✅", "DOCUMENTO AUTÊNTICO"
            color = "#22c55e"
        elif verdict.lower() == 'suspeito':
            css_class, icon, label = "suspect", "⚠️", "DOCUMENTO SUSPEITO"
            color = "#facc15"
        elif verdict.lower() == 'fraudulento':
            css_class, icon, label = "fraud", "❌", "DOCUMENTO FRAUDULENTO"
            color = "#ef4444"
        else:
            css_class, icon, label = "unknown", "ℹ️", f"Veredito: {verdict}"
            color = "#94a3b8"

        st.markdown(f"""
        <style>
        .verdict-card {{
            background-color: #1e293b;
            border-left: 6px solid {color};
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 0 6px rgba(0,0,0,0.3);
        }}
        .verdict-header {{
            font-size: 1.3rem;
            font-weight: 600;
            color: {color};
            margin-bottom: 0.3rem;
        }}
        .verdict-sub {{
            font-size: 1.05rem;
            color: #f1f5f9;
        }}
        </style>

        <div class="verdict-card">
            <div class="verdict-header">{icon} {label}</div>
            <div class="verdict-sub">Nível de confiança: <strong>{confidence}</strong></div>
        </div>
        """, unsafe_allow_html=True)

        justification = forensic.get('detailed_justification', 'Nenhuma justificativa fornecida.')
        st.markdown("### 📝 Justificativa Detalhada")
        st.markdown(f"""
        <div style='background-color:#1e293b;padding:1rem 1.2rem;border-radius:10px;border-left:5px solid {color};
                    box-shadow:0 0 6px rgba(0,0,0,0.3);color:#f1f5f9;'>
            {justification}
        </div>
        """, unsafe_allow_html=True)


    # --- TAB 2: Análise Detalhada ---
    with tab2:
        st.markdown("### 🔍 Análise Detalhada por Categoria")

        st.markdown("""
        <style>
        .section-box {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 0 6px rgba(0,0,0,0.3);
            color: #f1f5f9;
        }
        .section-title {
            font-weight: 600;
            margin-bottom: 0.3rem;
            font-size: 1.1rem;
        }
        .bg-box { border-left: 6px solid #3b82f6; }
        .sec-box { border-left: 6px solid #a855f7; }
        .dig-box { border-left: 6px solid #f97316; }
        </style>
        """, unsafe_allow_html=True)

        bg = forensic.get('background_analysis', {})
        sec = forensic.get('security_elements', {})
        dig = forensic.get('digital_forensics', {})

        # Fundo e textura
        st.markdown(f"""
        <div class="section-box bg-box">
            <div class="section-title">🎨 Fundo e Textura</div>
            <b>Consistência:</b> {bg.get('pattern_consistency','N/A')}<br>
            <b>Detalhes:</b> {bg.get('findings','N/A')}<br>
            <b>Evidências:</b> {', '.join(bg.get('evidence', [])) or 'Nenhuma'}<br>
        </div>
        """, unsafe_allow_html=True)

        # Elementos de segurança
        st.markdown(f"""
        <div class="section-box sec-box">
            <div class="section-title">🛡️ Elementos de Segurança</div>
            <b>Qualidade dos símbolos:</b> {sec.get('official_symbols_quality','N/A')}<br>
            <b>Detectados:</b> {', '.join(sec.get('detected_security_features', [])) or 'Nenhum'}<br>
            <b>Ausentes esperados:</b> {', '.join(sec.get('missing_expected_features', [])) or 'Nenhum'}<br>
            <b>Justificativa:</b> {sec.get('justification','N/A')}<br>
        </div>
        """, unsafe_allow_html=True)

        # Forense digital
        st.markdown(f"""
        <div class="section-box dig-box">
            <div class="section-title">💻 Forense Digital</div>
            <b>Compressão:</b> {dig.get('compression_artifacts','N/A')}<br>
            <b>Bordas:</b> {dig.get('edge_analysis','N/A')}<br>
            <b>Clonagem:</b> {dig.get('cloning_patterns','N/A')}<br>
            <b>Anomalias de pixel:</b> {', '.join(dig.get('pixel_level_anomalies', [])) or 'Nenhuma'}<br>
            <b>Justificativa:</b> {dig.get('justification','N/A')}<br>
        </div>
        """, unsafe_allow_html=True)


    # --- TAB 3 ---
    with tab3:
        anomalies = forensic.get('anomalies_detected', [])
        if anomalies:
            st.markdown(f"### ⚠️ {len(anomalies)} Anomalia(s) Detectada(s)")
            for i,a in enumerate(anomalies,1):
                sev = a.get('severity','médio')
                emoji = "🔴" if sev=='alto' else ("🟡" if sev=='médio' else "🟢")
                with st.expander(f"{emoji} Anomalia {i}: {a.get('type','Desconhecida')}"):
                    st.markdown(f"**Localização:** {a.get('location','N/A')}")
                    st.markdown(f"**Severidade:** {sev}")
                    st.markdown(f"**Evidência:** {a.get('evidence','N/A')}")
        else:
            st.success("✅ Nenhuma anomalia significativa detectada.")

    # --- TAB 4: Recomendações (Mensagem padrão) ---
    with tab4:
        st.markdown("### 💡 IMPORTANTE")

        st.markdown("""
        <style>
        .alert-box {
            background-color: #1e293b;
            border-left: 6px solid #3b82f6;
            border-radius: 10px;
            padding: 1.2rem 1.4rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0 6px rgba(0,0,0,0.3);
            color: #f1f5f9;
            font-size: 1.05rem;
            line-height: 1.6;
        }
        .alert-title {
            font-weight: 600;
            font-size: 1.15rem;
            color: #60a5fa;
            margin-bottom: 0.5rem;
        }
        </style>

        <div class="alert-box">
            <div class="alert-title">⚠️ Aviso Importante</div>
            Esta análise foi gerada automaticamente por um sistema de Inteligência Artificial 
            e <strong>pode conter erros</strong>. 
            Este relatório <strong>não substitui uma perícia oficial</strong>.
            <br><br>
            Em caso de suspeita de fraude, recomenda-se encaminhar o documento para uma 
            <strong>autoridade competente</strong> (ex.: delegacia, instituto de perícia ou cartório).
        </div>
        """, unsafe_allow_html=True)

        # Download JSON report button
        json_str = json.dumps({
            "pii_info": pii_info,
            "features": features,
            "forensic_result": forensic_result
        }, indent=2, ensure_ascii=False)

        st.download_button(
            label="📥 Baixar Relatório Completo (JSON)",
            data=json_str,
            file_name=f"forensic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )




# ==================== NOVAS FUNÇÕES ====================

def get_text_snippet(image_np: np.ndarray, box: List[float]) -> str:
    """
    Extrai o texto (OCR) de uma PII box para exibir na UI de revisão.
    """
    try:
        h, w = image_np.shape[:2]
        
        # Converte coordenadas normalizadas para pixels
        y1 = int(box[0] * h)
        x1 = int(box[1] * w)
        y2 = int(box[2] * h)
        x2 = int(box[3] * w)
        
        # Adiciona uma pequena margem para melhorar o OCR
        margin = 5
        y1 = max(0, y1 - margin)
        x1 = max(0, x1 - margin)
        y2 = min(h, y2 + margin)
        x2 = min(w, x2 + margin)

        if y2 <= y1 or x2 <= x1:
            return "[Caixa Inválida]"
        
        # Corta a região
        roi = image_np[y1:y2, x1:x2]
        
        # Converte para PIL Image para Tesseract
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        
        # Executa OCR
        # lang='por' para Português
        text = pytesseract.image_to_string(roi_pil, lang='por', config='--psm 7')
        
        text = text.strip().replace('\n', ' ')
        
        if not text:
            return "[Imagem/Foto]"
        
        # Limita o tamanho do snippet
        return f"'{text[:50]}{'...' if len(text) > 50 else ''}'"
        
    except Exception as e:
        st.error(f"Erro no Tesseract: {e}. Verifique se está instalado.")
        return "[Erro no OCR]"

def display_pii_review_ui():
    """
    (NOVA TELA) Exibe a UI para o usuário revisar as PII detectadas.
    """
    # Use the familiar left image + right review layout,
    # but render the PII checklist in two columns inside the right pane.
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('## 🛡️ Revisão de Informações Pessoais (PII)')
    st.markdown('Revise as informações detectadas pela IA local. Desmarque qualquer item que você **deseja manter visível** para a análise forense na nuvem.')
    st.markdown('**Padrão: Todas as PII são censuradas.**')

    pii_regions = st.session_state.get('pii_regions', [])
    original_image = st.session_state.get('original_image')

    if original_image is None or not pii_regions:
        st.error("Erro: Imagem original ou PII não encontradas no estado.")
        st.stop()

    # Left: image (smaller). Right: review form (with two internal columns)
    left, right = st.columns([1, 1])

    with left:
        st.markdown('### 🔍 Detecções da IA')
        debug_image = create_detection_debug_image(original_image, pii_regions)
        # keep image modest to avoid dominating the layout
        st.image(cv2.cvtColor(debug_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption('Todas as PII detectadas estão marcadas em verde.')

    with right:
        st.markdown('### 📋 Lista de PII Detectadas')

        with st.form(key="pii_review_form"):
            st.markdown('<div class="review-container">', unsafe_allow_html=True)

            # split list into two roughly equal halves
            n = len(pii_regions)
            mid = (n + 1) // 2
            left_list = pii_regions[:mid]
            right_list = pii_regions[mid:]

            c1, c2 = st.columns(2)

            def render_region(col, idx, region):
                key_snip = f"pii_snippet_{idx}"
                if key_snip not in st.session_state:
                    st.session_state[key_snip] = get_text_snippet(original_image, region['box'])
                snippet = st.session_state[key_snip]
                # show snippet with slightly larger font
                col.markdown(f"<div style='padding:6px;border-radius:6px;background:rgba(255,255,255,0.03);margin-bottom:8px'><strong>PII {idx+1}</strong><div style='font-family:monospace;color:#e6eef8;font-size:13px;margin-top:4px'>{snippet}</div></div>", unsafe_allow_html=True)
                col.checkbox(label="Censurar", value=True, key=f"censure_{region['id']}")

            for i, r in enumerate(left_list):
                render_region(c1, i, r)
            for j, r in enumerate(right_list, start=mid):
                render_region(c2, j, r)

            st.markdown('</div>', unsafe_allow_html=True)
            submit_review = st.form_submit_button("🔒 Confirmar Redações e Iniciar Análise", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if submit_review:
        # Lógica para processar o formulário
        final_pii_to_redact = []
        for region in pii_regions:
            # Verifica se o checkbox correspondente foi marcado
            if st.session_state[f"censure_{region['id']}"]:
                final_pii_to_redact.append(region)
        
        # Salva a lista final
        st.session_state['final_pii_to_redact'] = final_pii_to_redact
        
        # Atualiza o estado para rodar a análise
        st.session_state['app_state'] = 'running'
        
        # Limpa os snippets para a próxima execução
        for i in range(len(pii_regions)):
            del st.session_state[f"pii_snippet_{i}"]
            
        st.rerun() # Re-executa o script para ir para o estado 'running'

# ==================== MAIN APPLICATION ====================

def main():
    # Inicializa o estado da aplicação
    if 'app_state' not in st.session_state:
        st.session_state['app_state'] = 'start'

    # Header
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown('# 🔬 Sistema Inteligente de Análise Forense')
    st.markdown('**Arquitetura Híbrida: Local AI (PII Protection) + Cloud AI (Forensic Analysis)**')
    st.markdown('UFSC - INE5448 | Artur Luiz Rizzato & Toru Soda')
    st.markdown('</div>', unsafe_allow_html=True)

    # Sidebar: app name + description + quick usage; system status moved to compact footer
    with st.sidebar:
        st.markdown("## 🔬 Sistema Inteligente de Análise Forense")
        st.markdown("Pequena ferramenta híbrida: redaction local de PII + análise forense na nuvem de áreas seguras.")
        st.markdown("---")
        st.markdown("### Como usar")
        st.markdown("1. Faça upload da imagem do documento")
        st.markdown("2. Clique em 'Iniciar Detecção Local' para detectar PII")
        st.markdown("3. Revise as PII e confirme as redações")
        st.markdown("4. O sistema enviará as áreas seguras para análise forense na nuvem")
        st.markdown("---")
        
        # Add spacer before status box
        st.markdown("<div style='height:100px'></div>", unsafe_allow_html=True)
        
        # Status box with relative positioning and margin
        status_lines = []
        if PII_MODEL is not None:
            status_lines.append("✅ Modelo Local: best.pt carregado")
        else:
            status_lines.append("❌ Modelo Local: best.pt não encontrado")

        if Config.GOOGLE_API_KEY:
            status_lines.append("✅ Google Gemini configurado")
        else:
            status_lines.append("❌ Gemini API Key não encontrada")

        status_html = "<div style='position:relative;margin-top:auto;padding:8px;border-radius:8px;background:rgba(255,255,255,0.02);color:#cbd5e1;font-size:13px;opacity:0.9'>"
        status_html += "<strong style='font-size:13px'>⚙️ Status do Sistema</strong><br>"
        status_html += "<div style='margin-top:6px'>" + "<br>".join(status_lines) + "</div></div>"
        st.markdown(status_html, unsafe_allow_html=True)
    
    # ================================================================
    # ESTADO 1: START (Upload)
    # ================================================================
    if st.session_state['app_state'] == 'start':
        st.markdown("### 📤 Upload do Documento")
        
        uploaded_file = st.file_uploader(
            "Selecione a imagem do documento",
            type=['jpg', 'jpeg', 'png'],
            help="Formatos suportados: JPG, JPEG, PNG"
        )
        
        if uploaded_file:
            # Salva temporariamente
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = Path(tmp.name)
            
            # Carrega imagem
            original_image = cv2.imread(str(tmp_path))
            
            # Redimensiona se necessário
            h, w = original_image.shape[:2]
            if max(h, w) > Config.MAX_IMAGE_SIZE:
                scale = Config.MAX_IMAGE_SIZE / max(h, w)
                original_image = cv2.resize(original_image, None, fx=scale, fy=scale)
                cv2.imwrite(str(tmp_path), original_image)
            
            # Salva no estado da sessão
            st.session_state['tmp_path'] = str(tmp_path)
            st.session_state['original_image'] = original_image

            # Preview
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("#### 📸 Imagem Carregada")
                st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_container_width=True)
                file_size = len(uploaded_file.getvalue()) / 1024
                st.caption(f"Tamanho: {file_size:.1f} KB | Dimensões: {original_image.shape[1]}x{original_image.shape[0]}px")
            
            with col2:
                st.markdown("#### 🎯 Análise do Documento")
                
                # Botão de Análise
                st.markdown('<div class="process-card">', unsafe_allow_html=True)
                st.markdown("""
                **Fluxo de Análise:**
                1. 🖥️ Local AI detecta PII
                2. 👤 Usuário revisa e aprova
                3. ☁️ Cloud AI analisa áreas seguras
                """)
                
                model_ready = PII_MODEL is not None
                google_ready = Config.GOOGLE_API_KEY is not None
                
                # Botão de início
                analyze_hybrid = st.button(
                    "🔬 Iniciar Detecção Local",
                    type="primary",
                    use_container_width=True,
                    disabled=not (model_ready and google_ready)
                )
                
                if not model_ready:
                    st.error("❌ Modelo Local 'best.pt' não encontrado.")
                if not google_ready:
                    st.error("❌ Google API Key não configurada.")
                
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            
            if analyze_hybrid:
                # --- INÍCIO DA DETECÇÃO ---
                with st.spinner("🔍 STAGE 1: Detectando PII com Local AI..."):
                    pii_result = detect_pii_with_yolo(Path(st.session_state['tmp_path']))
                    
                    if not pii_result.get('success'):
                        st.error(f"❌ Erro na detecção de PII: {pii_result.get('error')}")
                    else:
                        # Salva resultados e muda o estado
                        st.session_state['pii_regions'] = pii_result['pii_regions']
                        st.session_state['app_state'] = 'review'
                        st.rerun() # Re-executa para mostrar a tela de revisão
            

    # ================================================================
    # ESTADO 2: REVIEW (Nova Tela)
    # ================================================================
    elif st.session_state['app_state'] == 'review':
        display_pii_review_ui()
        # O botão "Confirmar" dentro da UI de revisão mudará o estado para 'running'

    # ================================================================
    # ESTADO 3: RUNNING (Processamento)
    # ================================================================
    elif st.session_state['app_state'] == 'running':
        progress_container = st.empty()
        status_container = st.empty()
        
        with progress_container.container():
            progress_bar = st.progress(0, text="Iniciando Análise Híbrida...")
        
        try:
            original_image = st.session_state['original_image']
            tmp_path = Path(st.session_state['tmp_path'])
            all_pii_regions = st.session_state['pii_regions']
            final_pii_to_redact = st.session_state['final_pii_to_redact']

            # STAGE 1.5: Extração de Features
            status_container.info("📊 Extraindo features técnicas...")
            progress_bar.progress(25, text="📊 Extraindo features técnicas...")
            features = extract_visual_features(original_image)
            
            # STAGE 1.6: Aplicar Redação
            status_container.info("✂️ Aplicando redação cirúrgica de PII...")
            progress_bar.progress(50, text="✂️ Aplicando redação de PII...")
            
            # Cria imagem redacted (APENAS com as PIIs selecionadas)
            redacted_image = apply_redaction(original_image, final_pii_to_redact)
            
            # Salva imagem redacted
            redacted_path = tmp_path.parent / f"redacted_{tmp_path.name}"
            cv2.imwrite(str(redacted_path), redacted_image)
            
            # STAGE 2: Forensic Analysis
            status_container.info("☁️ STAGE 2: Enviando para análise forense cloud...")
            progress_bar.progress(75, text="☁️ Enviando para análise forense cloud...")
            
            # Prepara pii_info para o relatório
            pii_info_report = {
                "all_pii_regions": all_pii_regions, # Todas as detecções
                "redacted_pii_regions": final_pii_to_redact, # Apenas as redigidas
                "total_pii_found": len(all_pii_regions),
                "total_pii_redacted": len(final_pii_to_redact)
            }

            forensic_result = analyze_forensics_with_cloud(
                redacted_path,
                features,
                pii_info_report # Envia o novo objeto de info
            )
            progress_bar.progress(100, text="✅ Análise concluída!")
            
            status_container.success("✅ Análise concluída!")
            
            import time
            time.sleep(1)
            progress_container.empty()
            status_container.empty()
            
            # Salva resultados finais
            st.session_state['app_state'] = 'finished'
            st.session_state['redacted_image'] = redacted_image
            st.session_state['pii_info'] = pii_info_report
            st.session_state['features'] = features
            st.session_state['forensic_result'] = forensic_result
            
            # Limpa arquivos temporários
            redacted_path.unlink()
            tmp_path.unlink()

            # Limpa estado de revisão
            del st.session_state['pii_regions']
            del st.session_state['final_pii_to_redact']
            
            st.rerun() # Re-executa para mostrar os resultados
            
        except Exception as e:
            st.error(f"❌ Erro no processamento: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.session_state['app_state'] = 'start' # Reseta

    # ================================================================
    # ESTADO 4: FINISHED (Resultados)
    # ================================================================
    elif st.session_state['app_state'] == 'finished':
        display_hybrid_results(
            st.session_state['original_image'],
            st.session_state['redacted_image'],
            st.session_state['pii_info'],
            st.session_state['features'],
            st.session_state['forensic_result']
        )
        
        # Botão para recomeçar
        if st.button("⬅️ Analisar Outro Documento"):
            # Limpa todo o estado
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    # Garante que o tesseract pode ser encontrado (ajuste o caminho se necessário)
    # No WSL, pode ser desnecessário se estiver no PATH
    # No Windows, seria: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    main()
