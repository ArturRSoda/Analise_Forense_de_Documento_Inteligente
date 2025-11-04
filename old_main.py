#!/usr/bin/env python3
"""
MVP - Interface Web para Detecção de Anomalias em Documentos
Universidade Federal de Santa Catarina - INE5448

Autores: Artur Luiz Rizzato, Toru Soda
"""

import streamlit as st
import google.generativeai as genai
import json
import cv2
import os
from pathlib import Path
from datetime import datetime
from PIL import Image
import tempfile
import base64

# ==================== CONFIGURAÇÃO ====================

st.set_page_config(
    page_title="Detector de Anomalias em Documentos",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado (melhor contraste, tipografia e cards legíveis)
st.markdown(
    """
<style>
    /* Page background */
    .stApp {
        background: linear-gradient(180deg, #0f1724 0%, #111827 60%);
        color: #e6eef8;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    /* Central container card */
    .container-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border-radius: 14px;
        padding: 1.25rem;
        box-shadow: 0 6px 18px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.03);
    }

    /* Header */
    .main-title {
        text-align: left;
        color: #e6eef8;
        font-size: 2.1rem;
        font-weight: 800;
        margin-bottom: 0rem;
        letter-spacing: -0.5px;
    }

    .subtitle {
        text-align: left;
        color: #c7d2fe;
        font-size: 0.95rem;
        margin-top: 0.25rem;
        margin-bottom: 0.75rem;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1220, #071027);
        border-right: 1px solid rgba(255,255,255,0.03);
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stImage {
        color: #e6eef8 !important;
    }

    /* Cards (info / success / warning / danger) with dark theme */
    .info-card, .success-card, .warning-card, .danger-card {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 0.75rem;
        color: #071026;
        font-weight: 600;
    }
    .info-card {
        background: linear-gradient(90deg, rgba(139,92,246,0.15), rgba(99,102,241,0.10));
        border-left: 6px solid rgba(99,102,241,0.95);
        color: #eef2ff;
    }
    .success-card {
        background: linear-gradient(90deg, rgba(34,197,94,0.12), rgba(16,185,129,0.08));
        border-left: 6px solid rgba(34,197,94,0.95);
        color: #042814;
    }
    .warning-card {
        background: linear-gradient(90deg, rgba(250,204,21,0.12), rgba(245,158,11,0.06));
        border-left: 6px solid rgba(245,158,11,0.95);
        color: #2b1f00;
    }
    .danger-card {
        background: linear-gradient(90deg, rgba(248,113,113,0.10), rgba(239,68,68,0.06));
        border-left: 6px solid rgba(239,68,68,0.95);
        color: #2b0b0b;
    }

    /* Buttons: larger, high-contrast */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7c3aed 0%, #4f46e5 100%) !important;
        color: white !important;
        font-weight: 700;
        padding: 0.75rem 1rem !important;
        border-radius: 10px !important;
        border: none !important;
        box-shadow: 0 6px 18px rgba(79,70,229,0.25);
        transition: transform 0.12s ease-in-out;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(79,70,229,0.35);
    }

    /* File uploader area */
    .uploadedFile {
        border: 2px dashed rgba(255,255,255,0.06) !important;
        border-radius: 12px;
        padding: 1rem;
        background: rgba(255,255,255,0.01);
    }

    /* Metric values contrast */
    [data-testid="stMetricValue"], .stMetric {
        color: #e6eef8 !important;
    }

    /* Tabs content card */
    .tab-card {
        background: rgba(255,255,255,0.02);
        padding: 0.75rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.02);
    }

    /* Small text styling for captions */
    .caption-small {
        color: #c7d2fe;
        font-size: 0.85rem;
        margin-top: 0.25rem;
    }

    /* Ensure images have subtle border and rounded corners */
    .stImage img {
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.04);
    }

    /* Expander header contrast */
    .streamlit-expanderHeader {
        background-color: rgba(255,255,255,0.015);
        border-radius: 6px;
        color: #e6eef8;
        font-weight: 700;
    }

    /* Make pre formatted JSON readable */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #06b6d4 0%, #7dd3fc 100%) !important;
        color: #042f3e !important;
        font-weight: 700 !important;
    }

    /* Small responsive tweak for narrow screens */
    @media (max-width: 640px) {
        .main-title { font-size: 1.6rem; }
        .subtitle { font-size: 0.85rem; }
    }
</style>
""",
    unsafe_allow_html=True,
)

# Configuração do Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
MAX_DIMENSION = 2048
JPEG_QUALITY = 90

# ==================== FUNÇÕES AUXILIARES ====================

@st.cache_resource
def initialize_gemini():
    """Inicializa o Gemini (cached)."""
    if not GOOGLE_API_KEY:
        return False
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        return True
    except:
        return False

def preprocess_image(image_file) -> Path:
    """Processa a imagem enviada."""
    # Salva temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        tmp.write(image_file.getvalue())
        tmp_path = Path(tmp.name)
    
    # Redimensiona se necessário
    img = cv2.imread(str(tmp_path))
    h, w = img.shape[:2]
    
    if max(h, w) > MAX_DIMENSION:
        scale = MAX_DIMENSION / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    cv2.imwrite(str(tmp_path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    
    return tmp_path

def build_analysis_prompt() -> str:
    """Prompt otimizado."""
    return """Você é um especialista em análise forense de documentos brasileiros (RG, CNH, CPF, Certidões).

Analise esta imagem e retorne um JSON estruturado.

**FORMATO JSON:**

{
  "tipo_documento": "tipo identificado",
  "confianca_identificacao": "alta|média|baixa",
  "dados_extraidos": {
    "campos_principais": [
      {"campo": "NOME", "valor": "texto", "confianca": "alta|média|baixa"}
    ],
    "numeros_identificacao": [
      {"tipo": "CPF/RG", "numero": "valor"}
    ],
    "datas": [
      {"tipo": "nascimento/emissão", "data": "DD/MM/AAAA", "valida": true}
    ]
  },
  "analise_visual": {
    "qualidade_imagem": "excelente|boa|média|ruim",
    "elementos_seguranca": ["lista"],
    "possui_foto": true,
    "possui_assinatura": true
  },
  "analise_forense": {
    "autenticidade": "autêntico|suspeito|adulterado",
    "confianca": "alto|médio|baixo",
    "anomalias_visuais": [
      {"tipo": "tipo", "descricao": "descricao detalhado ", "severidade": "alta|média|baixa"}
    ],
    "anomalias_semanticas": [
      {"campo": "campo", "problema": "problema detalhado"}
    ],
    "pontos_suspeitos": [
      {"regiao": "região", "motivo": "motivo detalhado"}
    ]
  },
  "validacao_dados": {
    "cpf_valido": true,
    "datas_coerentes": true,
    "idade_compativel": true
  },
  "resumo_executivo": "Resumo completo da análise",
  "recomendacoes": ["lista de recomendações"]
}

Retorne APENAS o JSON, sem markdown."""

def analyze_document(image_path: Path) -> dict:
    """Analisa documento com Gemini."""
    img = Image.open(image_path)
    
    model = genai.GenerativeModel("gemini-2.5-flash")
    
    response = model.generate_content(
        [build_analysis_prompt(), img],
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            top_p=0.95,
            max_output_tokens=8192,
        )
    )
    
    # Parse JSON
    text = response.text
    text = text.replace('```json', '').replace('```', '').strip()
    
    # Tenta completar JSON se incompleto
    if not text.endswith('}'):
        open_braces = text.count('{')
        close_braces = text.count('}')
        if open_braces > close_braces:
            text += '}' * (open_braces - close_braces)
    
    return json.loads(text)

def get_emoji_autenticidade(autenticidade: str) -> str:
    """Retorna emoji baseado na autenticidade."""
    if autenticidade == "autêntico":
        return "✅"
    elif autenticidade == "suspeito":
        return "⚠️"
    else:
        return "❌"

def get_color_severidade(severidade: str) -> str:
    """Retorna cor baseada na severidade."""
    if severidade == "alta":
        return "#dc3545"
    elif severidade == "média":
        return "#ffc107"
    else:
        return "#28a745"

# ==================== INTERFACE ====================

def main():
    # Header (colocado em container-card para contraste)
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">🔍 Detector de Anomalias em Documentos</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análise Forense com Inteligência Artificial | UFSC - INE5448</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")  # espaçamento

    # Sidebar
    with st.sidebar:
        st.markdown('<div style="padding:0.5rem 0 0.75rem 0">', unsafe_allow_html=True)
        st.image("https://via.placeholder.com/300x100/0b1220/ffffff?text=UFSC+IA", use_container_width=True)
        st.markdown("### 📋 Informações do Projeto")
        st.markdown(
            """
            **Autores:**
            - Artur Luiz Rizzato
            - Toru Soda
            
            **Modelo:** Google Gemini 2.5 Flash
            
            **Documentos Suportados:**
            - 🪪 RG (Registro Geral)
            - 🚗 CNH (Carteira de Habilitação)
            - 📄 CPF
            - 📜 Certidões (Nascimento, Casamento)
            - 🛂 Outros documentos brasileiros
            """)
        st.markdown("---")
        st.markdown("### ⚙️ Status do Sistema")
        
        if GOOGLE_API_KEY:
            st.success("✅ API Key configurada")
        else:
            st.error("❌ API Key não encontrada")
            st.info("Configure: `export GOOGLE_API_KEY='sua_chave'`")
        st.markdown("</div>", unsafe_allow_html=True)

    # Verifica API
    if not initialize_gemini():
        st.error("⚠️ Erro ao conectar com Google Gemini. Verifique sua API Key.")
        st.stop()
    
    # Upload de arquivo (dentro de um card)
    st.markdown('<div class="container-card">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload do Documento")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem do documento (JPG, PNG, JPEG)",
        type=['jpg', 'jpeg', 'png'],
        help="Envie uma foto ou scan do documento a ser analisado"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file:
        # Usar container-card ao redor das colunas para contraste
        st.markdown('<div class="container-card">', unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Imagem Original")
            st.image(uploaded_file, use_container_width=True)
            file_size = len(uploaded_file.getvalue()) / 1024
            st.markdown(f'<p class="caption-small">Tamanho: {file_size:.1f} KB</p>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 🎯 Ações")
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown("""
            **Instruções rápidas:**
            1. Confira iluminação e foco
            2. Clique em "Analisar Documento"
            3. Resultado aparece em abas organizadas
            """)
            st.markdown('</div>', unsafe_allow_html=True)
            
            analyze_button = st.button("🔍 Analisar Documento", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Análise
        if analyze_button:
            with st.spinner("🔄 Processando imagem..."):
                try:
                    # Pré-processa
                    processed_path = preprocess_image(uploaded_file)
                    
                    # Barra de progresso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("📸 Imagem processada...")
                    progress_bar.progress(25)
                    
                    status_text.text("🤖 Enviando para IA...")
                    progress_bar.progress(50)
                    
                    # Analisa
                    analysis = analyze_document(processed_path)
                    
                    status_text.text("✅ Análise concluída!")
                    progress_bar.progress(100)
                    
                    # Limpa
                    import time
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Salva no session state
                    st.session_state['analysis'] = analysis
                    st.session_state['analyzed'] = True
                    
                    # Limpa arquivo temporário
                    processed_path.unlink()
                    
                except Exception as e:
                    st.error(f"❌ Erro na análise: {str(e)}")
                    st.stop()
    
    # Exibe resultados
    if st.session_state.get('analyzed', False):
        analysis = st.session_state['analysis']
        
        st.markdown("---")
        st.markdown('<div class="container-card">', unsafe_allow_html=True)
        st.markdown("## 📊 Resultados da Análise")
        
        # Métricas principais
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tipo = analysis.get('tipo_documento', 'N/A')
            st.metric("📄 Tipo de Documento", tipo)
        
        with col2:
            forense = analysis.get('analise_forense', {})
            autenticidade = forense.get('autenticidade', 'N/A')
            emoji = get_emoji_autenticidade(autenticidade)
            st.metric(f"{emoji} Autenticidade", autenticidade.upper())
        
        with col3:
            confianca = forense.get('confianca', 'N/A')
            st.metric("🎯 Confiança", confianca.upper())
        
        # Tabs para organizar informações (cada tab com tab-card)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📝 Dados Extraídos",
            "👁️ Análise Visual",
            "🔍 Análise Forense",
            "✅ Validação",
            "📋 Resumo"
        ])
        
        # Tab 1: Dados Extraídos
        with tab1:
            st.markdown('<div class="tab-card">', unsafe_allow_html=True)
            dados = analysis.get('dados_extraidos', {})
            
            st.markdown("### Campos Principais")
            campos = dados.get('campos_principais', [])
            if campos:
                for campo in campos:
                    col_a, col_b, col_c = st.columns([2, 3, 1])
                    with col_a:
                        st.markdown(f"**{campo.get('campo', 'N/A')}**")
                    with col_b:
                        st.text(campo.get('valor', 'N/A'))
                    with col_c:
                        conf = campo.get('confianca', 'N/A')
                        if conf == 'alta':
                            st.success(conf)
                        elif conf == 'média':
                            st.warning(conf)
                        else:
                            st.error(conf)
            else:
                st.info("Nenhum campo extraído.")
            
            st.markdown("### Números de Identificação")
            numeros = dados.get('numeros_identificacao', [])
            if numeros:
                for num in numeros:
                    st.info(f"**{num.get('tipo', 'N/A')}:** {num.get('numero', 'N/A')}")
            else:
                st.info("Nenhum número identificado.")
            
            st.markdown("### Datas")
            datas = dados.get('datas', [])
            if datas:
                for data in datas:
                    valida = "✅" if data.get('valida', False) else "❌"
                    st.text(f"{valida} {data.get('tipo', 'N/A').capitalize()}: {data.get('data', 'N/A')}")
            else:
                st.info("Nenhuma data extraída.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 2: Análise Visual
        with tab2:
            st.markdown('<div class="tab-card">', unsafe_allow_html=True)
            visual = analysis.get('analise_visual', {})
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("### Características")
                qualidade = visual.get('qualidade_imagem', 'N/A')
                st.info(f"**Qualidade:** {qualidade.upper()}")
                
                carac = visual.get('caracteristicas_visuais', {}) if 'caracteristicas_visuais' in visual else visual
                st.text(f"📸 Foto: {'Sim' if carac.get('possui_foto') else 'Não'}")
                st.text(f"✍️ Assinatura: {'Sim' if carac.get('possui_assinatura') else 'Não'}")
                st.text(f"📊 Código de Barras: {'Sim' if carac.get('possui_codigo_barras') else 'Não'}")
                st.text(f"📱 QR Code: {'Sim' if carac.get('possui_qr_code') else 'Não'}")
            
            with col_b:
                st.markdown("### Elementos de Segurança")
                elementos = visual.get('elementos_seguranca', []) or visual.get('elementos_seguranca_detectados', [])
                if elementos:
                    for elem in elementos:
                        st.success(f"✓ {elem}")
                else:
                    st.warning("Nenhum elemento de segurança detectado")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 3: Análise Forense
        with tab3:
            st.markdown('<div class="tab-card">', unsafe_allow_html=True)
            forense = analysis.get('analise_forense', {})
            
            # Status geral
            autenticidade = forense.get('autenticidade', 'N/A')
            if autenticidade == "autêntico":
                st.markdown('<div class="success-card">', unsafe_allow_html=True)
                st.markdown(f"### ✅ Documento aparenta ser AUTÊNTICO")
                st.markdown('</div>', unsafe_allow_html=True)
            elif autenticidade == "suspeito":
                st.markdown('<div class="warning-card">', unsafe_allow_html=True)
                st.markdown(f"### ⚠️ Documento SUSPEITO - requer verificação adicional")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="danger-card">', unsafe_allow_html=True)
                st.markdown(f"### ❌ Documento aparenta ser ADULTERADO")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Anomalias Visuais
            st.markdown("### ⚠️ Anomalias Visuais")
            anomalias_vis = forense.get('anomalias_visuais', [])
            if anomalias_vis:
                for anom in anomalias_vis:
                    severidade = anom.get('severidade', 'baixa')
                    with st.expander(f"🔴 {anom.get('tipo', 'Anomalia')} [{severidade.upper()}]"):
                        st.markdown(f"**Descrição:** {anom.get('descricao', 'N/A')}")
                        if 'localizacao' in anom:
                            st.markdown(f"**Localização:** {anom.get('localizacao', 'N/A')}")
            else:
                st.success("✅ Nenhuma anomalia visual detectada")
            
            # Anomalias Semânticas
            st.markdown("### 📊 Anomalias Semânticas")
            anomalias_sem = forense.get('anomalias_semanticas', [])
            if anomalias_sem:
                for anom in anomalias_sem:
                    st.warning(f"**{anom.get('campo', 'Campo')}:** {anom.get('problema', 'N/A')}")
            else:
                st.success("✅ Nenhuma anomalia semântica detectada")
            
            # Pontos Suspeitos
            st.markdown("### 🚨 Pontos Suspeitos")
            suspeitos = forense.get('pontos_suspeitos', [])
            if suspeitos:
                for ponto in suspeitos:
                    with st.expander(f"⚠️ {ponto.get('regiao', 'Região')}"):
                        st.markdown(f"**Motivo:** {ponto.get('motivo', 'N/A')}")
                        if 'recomendacao' in ponto:
                            st.info(f"💡 {ponto.get('recomendacao', '')}")
            else:
                st.success("✅ Nenhum ponto suspeito identificado")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 4: Validação
        with tab4:
            st.markdown('<div class="tab-card">', unsafe_allow_html=True)
            validacao = analysis.get('validacao_dados', {})
            
            st.markdown("### Validação de Dados")
            
            checks = [
                ("CPF válido", validacao.get('cpf_valido', False)),
                ("Datas coerentes", validacao.get('datas_coerentes', False)),
                ("Idade compatível", validacao.get('idade_compativel', False)),
                ("Formatação correta", validacao.get('formatacao_correta', False))
            ]
            
            col1, col2 = st.columns(2)
            
            for i, (label, value) in enumerate(checks):
                with col1 if i % 2 == 0 else col2:
                    if value:
                        st.success(f"✅ {label}")
                    else:
                        st.error(f"❌ {label}")
            
            # Recomendações
            st.markdown("### 💡 Recomendações")
            recs = analysis.get('recomendacoes', [])
            if recs:
                for i, rec in enumerate(recs, 1):
                    st.info(f"{i}. {rec}")
            else:
                st.success("Nenhuma recomendação adicional necessária")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Tab 5: Resumo
        with tab5:
            st.markdown('<div class="tab-card">', unsafe_allow_html=True)
            st.markdown("### 📋 Resumo Executivo")
            resumo = analysis.get('resumo_executivo', 'Resumo não disponível')
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown(resumo)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Metadados
            if 'metadados' in analysis:
                meta = analysis['metadados']
                st.markdown("### 📊 Estatísticas")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Campos Extraídos", meta.get('total_campos_extraidos', 0))
                with col2:
                    st.metric("Anomalias", meta.get('total_anomalias_encontradas', 0))
                with col3:
                    st.metric("Pontos Suspeitos", meta.get('total_pontos_suspeitos', 0))
            
            # Download JSON
            st.markdown("### 💾 Exportar Resultados")
            json_str = json.dumps(analysis, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar JSON Completo",
                data=json_str,
                file_name=f"analise_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
