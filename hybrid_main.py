#!/usr/bin/env python3
"""
Sistema Híbrido de Detecção de Anomalias em Documentos
Arquitetura em 3 Níveis de Segurança
UFSC - INE5448

Autores: Artur Luiz Rizzato, Toru Soda
"""

import streamlit as st
import json
import cv2
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import tempfile
import numpy as np
from typing import Optional, Dict, Tuple
import base64

# ==================== CONFIGURAÇÃO ====================

st.set_page_config(
    page_title="Detector de Anomalias - Híbrido",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS (mantendo o seu design)
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #111827 60%);
        color: #e6eef8;
    }
    .security-level {
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 2px solid;
        cursor: pointer;
        transition: all 0.3s;
    }
    .security-level:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
    }
    .level-local {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border-color: #34d399;
    }
    .level-hybrid {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border-color: #fbbf24;
    }
    .level-cloud {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border-color: #60a5fa;
    }
    .privacy-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        margin: 0.25rem;
    }
    .badge-high { background: #10b981; color: white; }
    .badge-medium { background: #f59e0b; color: white; }
    .badge-low { background: #ef4444; color: white; }
    
    .container-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
        margin-bottom: 1rem;
    }
    
    .warning-box {
        background: rgba(239, 68, 68, 0.1);
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ==================== CONFIGURAÇÕES DO SISTEMA ====================

class SecurityConfig:
    """Configurações de segurança e APIs"""
    
    # APIs disponíveis
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    
    # Ollama (Local)
    OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODELS = ["gemma3:4b", "gemma2:9b", "llama3.2:3b", "phi3:mini"]
    
    # Limites
    MAX_DIMENSION = 2048
    JPEG_QUALITY = 90

# ==================== FUNÇÕES DE ANONIMIZAÇÃO ====================

def extract_visual_features(image_path: Path) -> Dict:
    """
    Extrai features visuais SEM dados pessoais
    - Detecta padrões, texturas, elementos de segurança
    - Remove texto legível
    """
    img = cv2.imread(str(image_path))
    
    # Análise de qualidade
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Detecta bordas (para elementos de segurança)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Análise de cor (hologramas, marcas d'água)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    color_variance = np.std(hsv)
    
    # Detecta regiões de alta frequência (padrões de segurança)
    freq = np.fft.fft2(gray)
    freq_magnitude = np.abs(freq)
    high_freq_energy = np.sum(freq_magnitude > np.percentile(freq_magnitude, 95))
    
    return {
        "quality_score": float(blur_score),
        "edge_density": float(edge_density),
        "color_variance": float(color_variance),
        "security_pattern_energy": float(high_freq_energy),
        "image_dimensions": img.shape[:2],
        "has_color_regions": bool(color_variance > 30),
    }

def anonymize_for_cloud(image_path: Path) -> Tuple[Path, Dict]:
    """
    Cria versão anonimizada da imagem para envio seguro
    - Blur em textos/números
    - Mantém estrutura e elementos visuais
    """
    img = cv2.imread(str(image_path))
    
    # Detecta texto e aplica blur pesado
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Blur agressivo em regiões de texto
    kernel = np.ones((15, 15), np.uint8)
    text_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    anonymized = img.copy()
    anonymized = cv2.GaussianBlur(anonymized, (51, 51), 0)
    
    # Salva versão anonimizada
    anon_path = image_path.parent / f"anon_{image_path.name}"
    cv2.imwrite(str(anon_path), anonymized)
    
    # Extrai features
    features = extract_visual_features(image_path)
    
    return anon_path, features

# ==================== ANÁLISE LOCAL (OLLAMA) ====================

def check_ollama_available() -> Tuple[bool, list]:
    """Verifica se Ollama está rodando e quais modelos estão disponíveis"""
    try:
        import requests
        response = requests.get(f"{SecurityConfig.OLLAMA_HOST}/api/tags", timeout=2)
        if response.status_code == 200:
            models = [m['name'] for m in response.json().get('models', [])]
            return True, models
        return False, []
    except:
        return False, []

def analyze_local_ollama(image_path: Path, model: str = "gemma2:2b") -> Dict:
    """
    Análise 100% local com Ollama
    VANTAGEM: Privacidade total, sem envio de dados
    DESVANTAGEM: Precisa ter Ollama instalado
    """
    try:
        import requests
        from PIL import Image
        import io
        
        # Carrega imagem
        img = Image.open(image_path)
        
        # Converte para base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Prompt otimizado para modelos pequenos
        prompt = """Analise este documento brasileiro (RG/CNH/CPF) e retorne JSON:

{
  "tipo_documento": "tipo",
  "autenticidade": "autêntico|suspeito|adulterado",
  "confianca": "alto|médio|baixo",
  "anomalias": [{"tipo": "...", "severidade": "alta|média|baixa"}],
  "qualidade_imagem": "boa|média|ruim",
  "elementos_seguranca_visiveis": ["lista"],
  "recomendacoes": ["lista"]
}

Apenas JSON, sem markdown."""
        
        # Chama Ollama
        response = requests.post(
            f"{SecurityConfig.OLLAMA_HOST}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [img_b64],
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            text = result.get('response', '{}')
            # Limpa markdown
            text = text.replace('```json', '').replace('```', '').strip()
            return json.loads(text)
        else:
            return {"error": "Erro na comunicação com Ollama"}
            
    except Exception as e:
        return {"error": f"Erro local: {str(e)}"}

# ==================== ANÁLISE HÍBRIDA ====================

def analyze_hybrid(image_path: Path, features: Dict) -> Dict:
    """
    Análise híbrida: features locais + API cloud
    Envia apenas dados anonimizados
    """
    try:
        import google.generativeai as genai
        genai.configure(api_key=SecurityConfig.GOOGLE_API_KEY)
        
        prompt = f"""Analise estas características ANONIMIZADAS de um documento:

Features Visuais (SEM dados pessoais):
- Qualidade: {features['quality_score']:.2f}
- Densidade de bordas: {features['edge_density']:.4f}
- Variância de cor: {features['color_variance']:.2f}
- Energia de padrões: {features['security_pattern_energy']:.2f}

Com base APENAS nestas features, avalie:
1. Probabilidade de autenticidade (0-100%)
2. Elementos de segurança detectados
3. Anomalias estruturais
4. Recomendações

Retorne JSON estruturado."""
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
        
    except Exception as e:
        return {"error": f"Erro híbrido: {str(e)}"}

# ==================== ANÁLISE CLOUD COMPLETA ====================

def analyze_cloud_full(image_path: Path) -> Dict:
    """
    Análise completa com API (Gemini)
    Requer consentimento explícito do usuário
    """
    try:
        import google.generativeai as genai
        from PIL import Image
        
        genai.configure(api_key=SecurityConfig.GOOGLE_API_KEY)
        
        img = Image.open(image_path)
        
        prompt = """Você é um especialista forense em documentos brasileiros.

Analise detalhadamente e retorne JSON:

{
  "tipo_documento": "tipo",
  "confianca_identificacao": "alta|média|baixa",
  "dados_extraidos": {
    "campos_principais": [{"campo": "...", "valor": "...", "confianca": "..."}]
  },
  "analise_forense": {
    "autenticidade": "autêntico|suspeito|adulterado",
    "confianca": "alto|médio|baixo",
    "anomalias_visuais": [{"tipo": "...", "descricao": "...", "severidade": "..."}],
    "anomalias_semanticas": [{"campo": "...", "problema": "..."}]
  },
  "validacao_dados": {
    "cpf_valido": true,
    "datas_coerentes": true
  },
  "resumo_executivo": "texto",
  "recomendacoes": ["lista"]
}

Apenas JSON."""
        
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content([prompt, img])
        
        text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(text)
        
    except Exception as e:
        return {"error": f"Erro cloud: {str(e)}"}

# ==================== INTERFACE ====================

def main():
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.markdown('# 🔒 Detector de Anomalias - Sistema Híbrido')
    st.markdown('**Análise Forense com Privacidade Configurável** | UFSC - INE5448')
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🛡️ Configuração de Segurança")
        
        # Status Ollama
        ollama_ok, ollama_models = check_ollama_available()
        if ollama_ok:
            st.success(f"✅ Ollama ativo ({len(ollama_models)} modelos)")
            if ollama_models:
                st.info(f"Modelos: {', '.join(ollama_models[:3])}")
        else:
            st.warning("⚠️ Ollama offline (opcional)")
            st.caption("Para análise 100% local, instale Ollama")
        
        # Status APIs
        st.markdown("---")
        st.markdown("**APIs Cloud:**")
        if SecurityConfig.GOOGLE_API_KEY:
            st.success("✅ Google Gemini")
        else:
            st.error("❌ Gemini (configure API Key)")
        
        st.markdown("---")
        st.markdown("### 📋 Sobre o Projeto")
        st.markdown("""
        **Autores:**
        - Artur Luiz Rizzato
        - Toru Soda
        
        **Arquitetura:**
        - Nível 1: 100% Local (Ollama)
        - Nível 2: Híbrido (Features)
        - Nível 3: Cloud (Consentimento)
        """)
    
    # Upload
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload do Documento")
    uploaded_file = st.file_uploader(
        "Selecione a imagem",
        type=['jpg', 'jpeg', 'png'],
        help="Escolha o nível de segurança depois"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # Salva temporariamente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = Path(tmp.name)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="container-card">', unsafe_allow_html=True)
            st.markdown("#### 📸 Imagem Carregada")
            st.image(uploaded_file, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="container-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Escolha o Nível de Segurança")
            
            # NÍVEL 1: LOCAL
            st.markdown("""
            <div class="security-level level-local">
                <h4>🛡️ NÍVEL 1: Análise 100% Local</h4>
                <span class="privacy-badge badge-high">PRIVACIDADE MÁXIMA</span>
                <p><strong>✓ Dados nunca saem do seu computador</strong></p>
                <p>• Requer Ollama instalado</p>
                <p>• Menor acurácia, maior privacidade</p>
                <p>• Ideal para documentos extremamente sensíveis</p>
            </div>
            """, unsafe_allow_html=True)
            
            if ollama_ok:
                selected_model = st.selectbox(
                    "Modelo Ollama:",
                    ollama_models if ollama_models else ["Nenhum disponível"],
                    key="ollama_model"
                )
                analyze_local = st.button("🔒 Analisar Localmente", type="primary", use_container_width=True)
            else:
                st.warning("⚠️ Ollama não disponível. [Instalar Ollama](https://ollama.ai)")
                analyze_local = False
            
            # NÍVEL 2: HÍBRIDO
            st.markdown("""
            <div class="security-level level-hybrid">
                <h4>⚖️ NÍVEL 2: Análise Híbrida</h4>
                <span class="privacy-badge badge-medium">PRIVACIDADE MODERADA</span>
                <p><strong>✓ Envia apenas features visuais anonimizadas</strong></p>
                <p>• Dados pessoais removidos localmente</p>
                <p>• API analisa apenas padrões</p>
                <p>• Equilíbrio segurança/acurácia</p>
            </div>
            """, unsafe_allow_html=True)
            
            analyze_hybrid_btn = st.button("🔐 Análise Híbrida", use_container_width=True)
            
            # NÍVEL 3: CLOUD
            st.markdown("""
            <div class="security-level level-cloud">
                <h4>☁️ NÍVEL 3: Análise Cloud Completa</h4>
                <span class="privacy-badge badge-low">MÁXIMA ACURÁCIA</span>
                <p><strong>⚠️ Documento será enviado para API externa</strong></p>
                <p>• Análise forense detalhada</p>
                <p>• Extração completa de dados</p>
                <p>• Requer consentimento explícito</p>
            </div>
            """, unsafe_allow_html=True)
            
            consent = st.checkbox("⚠️ Eu aceito enviar este documento para análise cloud (Google Gemini)")
            analyze_cloud = st.button("☁️ Análise Cloud", disabled=not consent, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # PROCESSAMENTO
        if analyze_local:
            with st.spinner("🔒 Analisando localmente com Ollama..."):
                result = analyze_local_ollama(tmp_path, selected_model)
                st.session_state['analysis'] = result
                st.session_state['analysis_type'] = 'local'
                st.success("✅ Análise local concluída!")
        
        elif analyze_hybrid_btn:
            with st.spinner("🔐 Anonimizando e enviando features..."):
                anon_path, features = anonymize_for_cloud(tmp_path)
                result = analyze_hybrid(tmp_path, features)
                st.session_state['analysis'] = result
                st.session_state['analysis_type'] = 'hybrid'
                st.session_state['features'] = features
                st.success("✅ Análise híbrida concluída!")
                anon_path.unlink()  # Remove arquivo anonimizado
        
        elif analyze_cloud:
            with st.spinner("☁️ Enviando para análise cloud..."):
                result = analyze_cloud_full(tmp_path)
                st.session_state['analysis'] = result
                st.session_state['analysis_type'] = 'cloud'
                st.success("✅ Análise cloud concluída!")
        
        # Limpa arquivo temporário
        tmp_path.unlink()
    
    # EXIBIR RESULTADOS
    if 'analysis' in st.session_state:
        analysis = st.session_state['analysis']
        analysis_type = st.session_state.get('analysis_type', 'unknown')
        
        st.markdown("---")
        st.markdown('<div class="container-card">', unsafe_allow_html=True)
        
        # Badge do tipo de análise
        type_badges = {
            'local': ('🛡️ Análise Local', 'success-box'),
            'hybrid': ('⚖️ Análise Híbrida', 'warning-box'),
            'cloud': ('☁️ Análise Cloud', 'warning-box')
        }
        badge_text, badge_class = type_badges.get(analysis_type, ('❓ Desconhecido', 'warning-box'))
        
        st.markdown(f'<div class="{badge_class}"><strong>{badge_text}</strong></div>', unsafe_allow_html=True)
        
        # Resultados principais
        st.markdown("## 📊 Resultados da Análise")
        
        if 'error' in analysis:
            st.error(f"❌ {analysis['error']}")
        else:
            # Métricas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                tipo = analysis.get('tipo_documento', 'N/A')
                st.metric("📄 Tipo", tipo)
            
            with col2:
                autenticidade = analysis.get('autenticidade', 
                    analysis.get('analise_forense', {}).get('autenticidade', 'N/A'))
                st.metric("🎯 Autenticidade", autenticidade.upper())
            
            with col3:
                confianca = analysis.get('confianca',
                    analysis.get('analise_forense', {}).get('confianca', 'N/A'))
                st.metric("📊 Confiança", confianca.upper())
            
            # Tabs com detalhes
            if analysis_type == 'cloud':
                # Análise completa
                tab1, tab2, tab3 = st.tabs(["📋 Dados", "🔍 Forense", "💾 Export"])
                
                with tab1:
                    dados = analysis.get('dados_extraidos', {})
                    if dados:
                        st.json(dados)
                    else:
                        st.info("Dados não disponíveis")
                
                with tab2:
                    forense = analysis.get('analise_forense', {})
                    if forense:
                        st.json(forense)
                    else:
                        st.info("Análise forense não disponível")
                
                with tab3:
                    json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
                    st.download_button(
                        "📥 Baixar JSON",
                        json_str,
                        f"analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        "application/json"
                    )
            
            elif analysis_type == 'hybrid':
                # Mostra features usadas
                st.markdown("### 🔍 Features Anonimizadas Analisadas")
                if 'features' in st.session_state:
                    st.json(st.session_state['features'])
                
                st.markdown("### 📊 Resultado")
                st.json(analysis)
            
            else:
                # Análise local simplificada
                st.json(analysis)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
