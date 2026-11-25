import gzip
import json
import os
import re
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="IAprova — Aprendizagem Adaptativa Explicável (TRL 4)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

TAXONOMIA_FECHADA = [
    "Contabilidade Geral",
    "Contabilidade de Custos e Gerencial",
    "Contabilidade Aplicada ao Setor Público",
    "Auditoria e Perícia Contábil",
    "Legislação e Ética Profissional",
    "Análise de Balanços e Indicadores",
    "Matemática Financeira e Estatística",
    "Língua Portuguesa",
]

PESO_AREA_CFC = {
    "Contabilidade Geral": 0.40,
    "Auditoria e Perícia Contábil": 0.14,
    "Contabilidade de Custos e Gerencial": 0.12,
    "Legislação e Ética Profissional": 0.10,
    "Contabilidade Aplicada ao Setor Público": 0.08,
    "Língua Portuguesa": 0.06,
    "Matemática Financeira e Estatística": 0.05,
    "Análise de Balanços e Indicadores": 0.05,
}

ARQUIVO_LOGS = "logs_interacao_iaprova.csv"


# ==============================================================================
# FUNÇÕES DE CURADORIA E NORMALIZAÇÃO
# ==============================================================================
def mapear_taxonomia(disciplina_atual, sub_area_atual):
    disc = str(disciplina_atual or "").lower()
    sub = str(sub_area_atual or "").lower()

    if "português" in disc or "interpretação" in sub or "morfologia" in sub:
        return "Língua Portuguesa"
    elif "matemática" in disc or "estatística" in disc or "probabilidade" in sub:
        return "Matemática Financeira e Estatística"
    elif "pública" in disc or "setor público" in disc:
        return "Contabilidade Aplicada ao Setor Público"
    elif "auditoria" in disc or "perícia" in disc or "normas de auditoria" in sub:
        return "Auditoria e Perícia Contábil"
    elif (
        "tributário" in disc
        or "civil" in disc
        or "trabalho" in disc
        or "legislação" in disc
        or "ética" in disc
        or "ética" in sub
        or "código de ética" in sub
    ):
        return "Legislação e Ética Profissional"
    elif "custos" in disc or "custeio" in sub or "orçamento" in sub or "gerencial" in disc:
        return "Contabilidade de Custos e Gerencial"
    elif "análise de balanços" in disc or "balanços" in sub or "índices" in sub:
        return "Análise de Balanços e Indicadores"
    return "Contabilidade Geral"


def formatar_alternativas(raw_alts):
    """Normaliza alternativas nos formatos list[dict], dict ou list[str]."""
    if not raw_alts:
        return []
    formatadas = []
    if isinstance(raw_alts, dict):
        for k, v in raw_alts.items():
            formatadas.append(f"{str(k).strip().rstrip(')')}) {str(v).strip()}")
    elif isinstance(raw_alts, list):
        for i, item in enumerate(raw_alts):
            letra_padrao = chr(65 + i)
            if isinstance(item, dict):
                label = str(item.get("label", letra_padrao)).strip().rstrip(")")
                txt = str(item.get("text", "")).strip()
                if txt:
                    formatadas.append(f"{label}) {txt}")
            else:
                txt = str(item).strip()
                if txt:
                    if not re.match(r"^[\(\[]?[A-E][\)\]\.\-]", txt):
                        txt = f"{letra_padrao}) {txt}"
                    formatadas.append(txt)
    return formatadas


def normalizar_json_para_df(dados_banco):
    registros = []
    contador = 0
    for exame in dados_banco:
        ano = exame.get("ano")
        versao = exame.get("versao")
        if ano and versao:
            nome_exame = f"CFC {ano}.{versao}"
        else:
            nome_exame = str(exame.get("exame", exame.get("prova", "Exame CFC")))
        origem = str(exame.get("origem", "CFC"))

        for q in exame.get("questoes", []):
            revisao = q.get("revisao_uso")
            if isinstance(revisao, dict) and revisao.get("status") == "bloqueada":
                continue

            resp = str(q.get("resposta", "")).strip().upper()
            if resp == "ANULADA":
                continue

            texto = str(q.get("question_text", "")).strip()
            if not texto:
                continue

            area = q.get("sub_area_normalizada")
            if area not in TAXONOMIA_FECHADA:
                area = mapear_taxonomia(q.get("disciplina"), q.get("sub_area"))

            alts = formatar_alternativas(q.get("options") or q.get("alternativas"))

            contador += 1
            registros.append({
                "id": f"{nome_exame}_Q{q.get('question_number', contador)}_{contador}",
                "exame": nome_exame,
                "origem": origem,
                "numero": q.get("question_number", contador),
                "texto": texto,
                "alternativas": alts,
                "resposta": resp,
                "area": area,
                "sub_area_original": str(q.get("sub_area", "")),
                "possui_gabarito": resp in ["A", "B", "C", "D", "E"],
            })
    return pd.DataFrame(registros)


def base_demonstracao_embutida():
    exemplos = [
        ("Contabilidade Geral", "CFC Demo", 1,
         "De acordo com a NBC TG 01 (Redução ao Valor Recuperável de Ativos), uma perda por desvalorização (impairment) deve ser reconhecida imediatamente no resultado quando:",
         ["A) O valor justo do ativo for superior ao valor contábil.", "B) O valor contábil do ativo exceder seu valor recuperável.", "C) O valor em uso for igual ao custo histórico.", "D) A depreciação acumulada atingir 50% do custo de aquisição."], "B"),
        ("Contabilidade de Custos e Gerencial", "CFC Demo", 2,
         "Uma indústria apresentou preço de venda unitário de R$ 100,00, custos variáveis unitários de R$ 60,00 e custos fixos totais de R$ 80.000,00. O Ponto de Equilíbrio Contábil (em unidades) é:",
         ["A) 1.000 unidades.", "B) 1.500 unidades.", "C) 2.000 unidades.", "D) 2.500 unidades."], "C"),
        ("Contabilidade Aplicada ao Setor Público", "CFC Demo", 3,
         "Conforme a Lei nº 4.320/1964 e o MCASP, os estágios da execução da despesa orçamentária ocorrem na seguinte ordem:",
         ["A) Liquidação, Empenho e Pagamento.", "B) Empenho, Liquidação e Pagamento.", "C) Previsão, Arrecadação e Recolhimento.", "D) Dotação, Empenho e Restos a Pagar."], "B"),
        ("Auditoria e Perícia Contábil", "CFC Demo", 4,
         "Segundo a NBC TA 705, quando o auditor conclui que as distorções são relevantes E generalizadas nas demonstrações contábeis, deve emitir:",
         ["A) Opinião com ressalva.", "B) Abstenção de opinião.", "C) Opinião sem ressalva com ênfase.", "D) Opinião adversa."], "D"),
        ("Legislação e Ética Profissional", "CFC Demo", 5,
         "De acordo com o Código de Ética Profissional do Contador (NBC PG 01), constitui dever fundamental do profissional:",
         ["A) Reter livros contábeis para cobrar honorários.", "B) Guardar sigilo sobre o que souber em razão do exercício profissional, salvo casos previstos em lei.", "C) Assinar demonstrações elaboradas por terceiros sem revisão.", "D) Aviltar honorários profissionais."], "B"),
        ("Análise de Balanços e Indicadores", "CFC Demo", 6,
         "Com Ativo Circulante de R$ 300.000, Estoques de R$ 100.000 e Passivo Circulante de R$ 200.000, os índices de Liquidez Corrente e Liquidez Seca são, respectivamente:",
         ["A) 1,50 e 1,00.", "B) 1,00 e 1,50.", "C) 2,00 e 1,00.", "D) 1,50 e 0,50."], "A"),
        ("Matemática Financeira e Estatística", "CFC Demo", 7,
         "Um capital de R$ 10.000,00 aplicado a juros compostos de 10% ao ano durante 2 anos gera o montante de:",
         ["A) R$ 12.000,00.", "B) R$ 12.100,00.", "C) R$ 11.000,00.", "D) R$ 12.500,00."], "B"),
        ("Língua Portuguesa", "CFC Demo", 8,
         "Assinale a alternativa em que a concordância verbal está inteiramente de acordo com a norma-padrão:",
         ["A) Fazem dois anos que o balanço foi publicado.", "B) Houveram divergências na assembleia.", "C) Aluga-se salas comerciais.", "D) Existem normas específicas para o relatório contábil."], "D"),
    ]
    return pd.DataFrame([
        {
            "id": f"DEMO_{i+1}", "exame": ex, "origem": "Demo", "numero": num,
            "texto": txt, "alternativas": alts, "resposta": resp,
            "area": area, "sub_area_original": area, "possui_gabarito": True,
        }
        for i, (area, ex, num, txt, alts, resp) in enumerate(exemplos)
    ])


@st.cache_data(show_spinner=False)
def carregar_base_automatica():
    pasta_app = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(pasta_app, "Banco_IAprova_Expandido.json.gz"),
        os.path.join(pasta_app, "Banco_IAprova_Expandido.json"),
        os.path.join(pasta_app, "Banco_IAprova_Taxonomia_Fechada.json"),
        os.path.join(pasta_app, "Banco_IAprova_Conferido.json"),
        r"E:\TCC\IAprova_App\Banco_IAprova_Expandido.json",
        r"E:\TCC\IAprova_App\Banco_IAprova_Conferido.json",
        r"E:\TCC\Banco_IAprova_Expandido.json",
        r"E:\TCC\Banco_IAprova_Conferido.json",
    ]
    for caminho in candidatos:
        if os.path.exists(caminho):
            if caminho.endswith(".gz"):
                with gzip.open(caminho, "rt", encoding="utf-8") as f:
                    dados = json.load(f)
            else:
                with open(caminho, "r", encoding="utf-8") as f:
                    dados = json.load(f)
            df = normalizar_json_para_df(dados)
            if not df.empty:
                return df, os.path.basename(caminho)
    return base_demonstracao_embutida(), "Modo Demonstração (8 itens)"


# ==============================================================================
# CLASSIFICADOR SEMÂNTICO LOCAL (TF-IDF N-GRAMAS + REGRESSÃO LOGÍSTICA)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def treinar_classificador_local(textos_tuple, areas_tuple):
    vec = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )
    X = vec.fit_transform(textos_tuple)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=1500,
        C=1.5,
        random_state=42,
    )
    clf.fit(X, areas_tuple)
    return vec, clf


def explicar_classificacao_texto(texto, vec, clf, top_k=6):
    X_q = vec.transform([texto])
    probs = clf.predict_proba(X_q)[0]
    classes = clf.classes_
    ordem = np.argsort(probs)[::-1]
    classe_pred = classes[ordem[0]]
    confianca = probs[ordem[0]]

    idx_classe = list(classes).index(classe_pred)
    pesos = clf.coef_[idx_classe]
    nomes_features = vec.get_feature_names_out()
    ativos = X_q.nonzero()[1]

    contribuicoes = []
    for idx_feat in ativos:
        score = float(X_q[0, idx_feat] * pesos[idx_feat])
        if score > 0:
            contribuicoes.append((nomes_features[idx_feat], score))
    contribuicoes.sort(key=lambda x: x[1], reverse=True)

    ranking_probs = pd.DataFrame({
        "Área (Taxonomia Fechada)": [classes[i] for i in ordem],
        "Probabilidade (%)": [round(float(probs[i]) * 100, 1) for i in ordem],
    })
    return classe_pred, float(confianca), ranking_probs, contribuicoes[:top_k]


# ==============================================================================
# MOTOR DE RECOMENDAÇÃO EXPLICÁVEL E LOGS
# ==============================================================================
def registrar_log(etapa, questao_row, resposta_usuario, acertou, explicacao):
    registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sessao_id": st.session_state.sessao_id,
        "etapa": etapa,
        "questao_id": questao_row["id"],
        "exame": questao_row["exame"],
        "area_taxonomia": questao_row["area"],
        "resposta_usuario": resposta_usuario,
        "gabarito": questao_row["resposta"],
        "acertou": int(acertou),
        "justificativa_pedagogica": explicacao,
    }
    st.session_state.logs.append(registro)

    try:
        df_novo = pd.DataFrame([registro])
        cabecalho = not os.path.exists(ARQUIVO_LOGS)
        df_novo.to_csv(ARQUIVO_LOGS, mode="a", header=cabecalho, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def calcular_perfil_estudante(logs):
    if not logs:
        return pd.DataFrame()

    df_logs = pd.DataFrame(logs)
    resumo = []
    for area in TAXONOMIA_FECHADA:
        sub = df_logs[df_logs["area_taxonomia"] == area]
        total = len(sub)
        acertos = int(sub["acertou"].sum()) if total > 0 else 0
        taxa = (acertos / total) if total > 0 else None
        peso = PESO_AREA_CFC.get(area, 0.05)

        if taxa is None:
            status = "Não Avaliada (Cold Start)"
            prioridade = 0.55 + peso * 0.30
        elif taxa < 0.50:
            status = "Lacuna Crítica"
            prioridade = (1.0 - taxa) * 0.70 + peso * 0.30
        elif taxa < 0.80:
            status = "Reforço Moderado"
            prioridade = (1.0 - taxa) * 0.70 + peso * 0.30
        else:
            status = "Consolidação"
            prioridade = (1.0 - taxa) * 0.70 + peso * 0.30

        resumo.append({
            "Área": area,
            "Respondidas": total,
            "Acertos": acertos,
            "Aproveitamento (%)": round(taxa * 100, 1) if taxa is not None else np.nan,
            "Peso CFC (%)": round(peso * 100, 1),
            "Status Pedagógico": status,
            "Índice de Prioridade": round(prioridade, 3),
        })

    return pd.DataFrame(resumo).sort_values("Índice de Prioridade", ascending=False).reset_index(drop=True)


def gerar_explicacao_pedagogica(linha_perfil, area_incidencia_base):
    area = linha_perfil["Área"]
    resp = int(linha_perfil["Respondidas"])
    acertos = int(linha_perfil["Acertos"])
    status = linha_perfil["Status Pedagógico"]
    peso_cfc = linha_perfil["Peso CFC (%)"]
    prioridade = linha_perfil["Índice de Prioridade"]

    if resp == 0:
        return (
            f"Recomendada pela regra de cobertura de **Cold Start**: você ainda não respondeu "
            f"questões de **{area}** (peso estimado de {peso_cfc:.0f}% no CFC e {area_incidencia_base:.1f}% na base ativa). "
            f"Índice de prioridade: **{prioridade:.3f}**."
        )
    taxa = (acertos / resp) * 100
    return (
        f"Recomendada sob a regra de **{status}**: seu aproveitamento atual em **{area}** é de "
        f"**{taxa:.0f}% ({acertos}/{resp})**, e essa área representa **{peso_cfc:.0f}%** do peso pedagógico do exame "
        f"({area_incidencia_base:.1f}% na base ativa). Fórmula rastreável: "
        f"Prioridade = 0,70 × (1 − acerto) + 0,30 × peso = **{prioridade:.3f}**."
    )


def extrair_opcoes_disponiveis(row):
    letras_padrao = ["A", "B", "C", "D"]
    if row["resposta"] == "E" or len(row["alternativas"]) >= 5:
        letras_padrao.append("E")
    elif re.search(r"(?:^|\n|\s)[\(\[]?E[\)\]\.\-]\s", row["texto"]):
        letras_padrao.append("E")
    return letras_padrao


def renderizar_questao(row):
    st.markdown(f"**Exame:** `{row['exame']}` &nbsp;|&nbsp; **Questão:** `{row['numero']}` &nbsp;|&nbsp; **Área (Taxonomia):** `{row['area']}`")
    with st.container(border=True):
        st.markdown(row["texto"])
    if row["alternativas"]:
        for alt in row["alternativas"]:
            st.markdown(f"- {alt}")


# ==============================================================================
# ESTADO DA SESSÃO
# ==============================================================================
if "sessao_id" not in st.session_state:
    st.session_state.sessao_id = str(uuid.uuid4())[:8].upper()
if "logs" not in st.session_state:
    st.session_state.logs = []
if "diagnostico_ids" not in st.session_state:
    st.session_state.diagnostico_ids = []
if "diagnostico_idx" not in st.session_state:
    st.session_state.diagnostico_idx = 0
if "diagnostico_concluido" not in st.session_state:
    st.session_state.diagnostico_concluido = False
if "rec_atual_id" not in st.session_state:
    st.session_state.rec_atual_id = None
if "rec_feedback" not in st.session_state:
    st.session_state.rec_feedback = None


# ==============================================================================
# BARRA LATERAL
# ==============================================================================
with st.sidebar:
    st.title("🎓 IAprova")
    st.caption("Protótipo Funcional — TRL 4 | UNIFAP Digital")
    st.divider()

    arquivo_upload = st.file_uploader(
        "Carregar base JSON (opcional)",
        type=["json"],
        help="Substitua a base ativa enviando Banco_IAprova_Conferido.json ou Banco_IAprova_Expandido.json.",
    )

    if arquivo_upload is not None:
        dados_up = json.load(arquivo_upload)
        df_base = normalizar_json_para_df(dados_up)
        fonte_ativa = f"Upload: {arquivo_upload.name}"
    else:
        df_base, fonte_ativa = carregar_base_automatica()

    df_com_gabarito = df_base[df_base["possui_gabarito"]].reset_index(drop=True)

    st.success(f"**Base ativa:** {fonte_ativa}")
    st.metric("Questões válidas (IA)", f"{len(df_base):,}".replace(",", "."))
    st.metric("Com gabarito (Simulados)", f"{len(df_com_gabarito):,}".replace(",", "."))
    st.caption(f"Sessão atual: `{st.session_state.sessao_id}`")

    st.divider()
    if st.button("🔄 Reiniciar Sessão de Teste", use_container_width=True):
        st.session_state.logs = []
        st.session_state.diagnostico_ids = []
        st.session_state.diagnostico_idx = 0
        st.session_state.diagnostico_concluido = False
        st.session_state.rec_atual_id = None
        st.session_state.rec_feedback = None
        st.rerun()

vec_local, clf_local = treinar_classificador_local(
    tuple(df_base["texto"].tolist()),
    tuple(df_base["area"].tolist()),
)


# ==============================================================================
# CABEÇALHO E ABAS
# ==============================================================================
st.title("IAprova — Classificação Semântica e Recomendação Explicável")
st.markdown(
    "Plataforma de aprendizagem adaptativa voltada ao **Exame de Suficiência do CFC** e concursos contábeis, "
    "combinando **PLN em português**, **taxonomia fechada (8 áreas)** e **regras pedagógicas rastreáveis (XAI)**."
)

aba_diag, aba_rec, aba_dash, aba_clf, aba_logs = st.tabs([
    "1️⃣ Simulado Diagnóstico (Cold Start)",
    "2️⃣ Recomendação Explicável",
    "3️⃣ Dashboard & Benchmark DSR",
    "4️⃣ Classificador Semântico (IA)",
    "5️⃣ Logs & Auditoria",
])


# ==============================================================================
# ABA 1: SIMULADO DIAGNÓSTICO
# ==============================================================================
with aba_diag:
    st.subheader("Simulado Diagnóstico Inicial — Mitigação de Cold Start")
    st.write(
        "Para estudantes sem histórico prévio, o sistema seleciona **1 questão por grande área da taxonomia fechada** "
        "(8 questões) para diagnosticar lacunas de conhecimento imediatamente."
    )

    if not st.session_state.diagnostico_ids:
        if st.button("▶️ Iniciar Simulado Diagnóstico (8 áreas)", type="primary"):
            selecionadas = []
            for area in TAXONOMIA_FECHADA:
                candidatas = df_com_gabarito[df_com_gabarito["area"] == area]
                if not candidatas.empty:
                    selecionadas.append(candidatas.sample(1).iloc[0]["id"])
            st.session_state.diagnostico_ids = selecionadas
            st.session_state.diagnostico_idx = 0
            st.session_state.diagnostico_concluido = False
            st.rerun()
    elif not st.session_state.diagnostico_concluido:
        idx = st.session_state.diagnostico_idx
        total_diag = len(st.session_state.diagnostico_ids)
        q_id = st.session_state.diagnostico_ids[idx]
        row_q = df_com_gabarito[df_com_gabarito["id"] == q_id].iloc[0]

        st.progress(idx / total_diag, text=f"Questão {idx + 1} de {total_diag}")
        renderizar_questao(row_q)

        opcoes = extrair_opcoes_disponiveis(row_q)
        escolha = st.radio("Selecione sua resposta:", opcoes, horizontal=True, key=f"diag_radio_{q_id}")

        if st.button("Confirmar Resposta e Avançar", type="primary", key=f"btn_diag_{q_id}"):
            acertou = escolha == row_q["resposta"]
            expl = f"Diagnóstico inicial (Cold Start): mapeamento de proficiência base na área {row_q['area']}."
            registrar_log("Simulado Diagnóstico", row_q, escolha, acertou, expl)

            if idx + 1 < total_diag:
                st.session_state.diagnostico_idx += 1
            else:
                st.session_state.diagnostico_concluido = True
            st.rerun()
    else:
        st.success("✅ Simulado Diagnóstico concluído! Seu perfil inicial de lacunas foi calculado.")
        df_perfil = calcular_perfil_estudante(st.session_state.logs)
        st.dataframe(df_perfil, use_container_width=True, hide_index=True)
        st.info("👉 Acesse a aba **2️⃣ Recomendação Explicável** para praticar nas áreas prioritárias com justificativa pedagógica.")


# ==============================================================================
# ABA 2: RECOMENDAÇÃO EXPLICÁVEL
# ==============================================================================
with aba_rec:
    st.subheader("Trilha Adaptativa com Explicabilidade Pedagógica (XAI)")

    df_perfil = calcular_perfil_estudante(st.session_state.logs)
    respondidas_ids = {l["questao_id"] for l in st.session_state.logs}

    if df_perfil.empty:
        st.warning(
            "Você ainda não respondeu ao **Simulado Diagnóstico (Aba 1)**. "
            "Enquanto isso, a recomendação abaixo segue a regra inicial de cobertura por peso do edital do CFC."
        )
        df_perfil = pd.DataFrame([
            {
                "Área": area,
                "Respondidas": 0,
                "Acertos": 0,
                "Aproveitamento (%)": np.nan,
                "Peso CFC (%)": round(PESO_AREA_CFC[area] * 100, 1),
                "Status Pedagógico": "Não Avaliada (Cold Start)",
                "Índice de Prioridade": round(0.55 + PESO_AREA_CFC[area] * 0.30, 3),
            }
            for area in TAXONOMIA_FECHADA
        ]).sort_values("Índice de Prioridade", ascending=False).reset_index(drop=True)

    linha_escolhida = None
    questao_rec = None

    if st.session_state.rec_atual_id and st.session_state.rec_atual_id not in respondidas_ids:
        matches = df_com_gabarito[df_com_gabarito["id"] == st.session_state.rec_atual_id]
        if not matches.empty:
            questao_rec = matches.iloc[0]
            linha_escolhida = df_perfil[df_perfil["Área"] == questao_rec["area"]].iloc[0]

    if questao_rec is None:
        for _, row_p in df_perfil.iterrows():
            disponiveis = df_com_gabarito[
                (df_com_gabarito["area"] == row_p["Área"])
                & (~df_com_gabarito["id"].isin(respondidas_ids))
            ]
            if not disponiveis.empty:
                questao_rec = disponiveis.sample(1).iloc[0]
                linha_escolhida = row_p
                st.session_state.rec_atual_id = questao_rec["id"]
                break

    if st.session_state.rec_feedback:
        tipo_fb, msg_fb = st.session_state.rec_feedback
        if tipo_fb == "ok":
            st.success(msg_fb)
        else:
            st.error(msg_fb)

    if questao_rec is not None and linha_escolhida is not None:
        incidencia_area = (df_base["area"] == linha_escolhida["Área"]).mean() * 100
        justificativa = gerar_explicacao_pedagogica(linha_escolhida, incidencia_area)

        with st.container(border=True):
            st.markdown("#### 🔍 Por que o IAprova recomendou esta questão? (XAI)")
            st.markdown(justificativa)

        renderizar_questao(questao_rec)
        opcoes_rec = extrair_opcoes_disponiveis(questao_rec)
        resp_rec = st.radio("Sua resposta:", opcoes_rec, horizontal=True, key=f"rec_radio_{questao_rec['id']}")

        col_b1, col_b2 = st.columns([1, 1])
        with col_b1:
            if st.button("✅ Responder e Recalcular Trilha", type="primary", use_container_width=True):
                acertou = resp_rec == questao_rec["resposta"]
                registrar_log("Recomendação Explicável", questao_rec, resp_rec, acertou, justificativa)
                if acertou:
                    st.session_state.rec_feedback = (
                        "ok",
                        f"Acertou! Gabarito oficial: {questao_rec['resposta']}. Perfil em {questao_rec['area']} atualizado.",
                    )
                else:
                    st.session_state.rec_feedback = (
                        "err",
                        f"Você marcou {resp_rec}; o gabarito oficial é {questao_rec['resposta']}. {questao_rec['area']} continua prioritária.",
                    )
                st.session_state.rec_atual_id = None
                st.rerun()
        with col_b2:
            if st.button("⏭️ Sortear Outra Questão Prioritária", use_container_width=True):
                st.session_state.rec_atual_id = None
                st.session_state.rec_feedback = None
                st.rerun()
    else:
        st.success("🎉 Você já respondeu todas as questões com gabarito disponíveis nesta sessão!")


# ==============================================================================
# ABA 3: DASHBOARD & BENCHMARK DSR
# ==============================================================================
with aba_dash:
    st.subheader("Learning Analytics — Desempenho do Estudante")

    if st.session_state.logs:
        df_l = pd.DataFrame(st.session_state.logs)
        df_perf = calcular_perfil_estudante(st.session_state.logs)
        avaliadas = df_perf[df_perf["Respondidas"] > 0]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Questões Respondidas", len(df_l))
        c2.metric("Acurácia Global", f"{df_l['acertou'].mean() * 100:.1f}%")
        c3.metric("Áreas em Lacuna Crítica", int((avaliadas["Status Pedagógico"] == "Lacuna Crítica").sum()))
        melhor_area = avaliadas.sort_values("Aproveitamento (%)", ascending=False).iloc[0]["Área"] if not avaliadas.empty else "-"
        c4.metric("Maior Domínio Atual", melhor_area)

        col_g1, col_g2 = st.columns([1.2, 1])
        with col_g1:
            st.markdown("**Aproveitamento por Área (%)**")
            st.bar_chart(avaliadas.set_index("Área")[["Aproveitamento (%)"]])
        with col_g2:
            st.markdown("**Matriz de Priorização Pedagógica (Rastreável)**")
            st.dataframe(
                df_perf[["Área", "Respondidas", "Aproveitamento (%)", "Status Pedagógico", "Índice de Prioridade"]],
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info("Responda ao menos uma questão nas abas 1 ou 2 para visualizar seu painel individual de Learning Analytics.")

    st.divider()
    st.subheader("Benchmark Científico do Projeto (Metodologia DSR — Gold Standard CFC)")
    df_bench = pd.DataFrame([
        {"Modelo Avaliado": "Transformer (BERTimbau — CFC)", "Acurácia (%)": 68.0, "Macro-F1 (%)": 45.0, "Acurácia Balanceada (%)": 56.5, "Custo de API": "Zero (Local)"},
        {"Modelo Avaliado": "Baseline TF-IDF (CFC)", "Acurácia (%)": 73.0, "Macro-F1 (%)": 52.0, "Acurácia Balanceada (%)": 61.4, "Custo de API": "Zero (Local)"},
        {"Modelo Avaliado": "TF-IDF Avançado N-gramas (CFC)", "Acurácia (%)": 75.0, "Macro-F1 (%)": 55.0, "Acurácia Balanceada (%)": 62.1, "Custo de API": "Zero (Local)"},
        {"Modelo Avaliado": "LLM Few-Shot (Qwen / Groq)", "Acurácia (%)": 78.0, "Macro-F1 (%)": 76.0, "Acurácia Balanceada (%)": 75.6, "Custo de API": "API Externa"},
        {"Modelo Avaliado": "TF-IDF + Expansão de Domínio (7.735 itens)", "Acurácia (%)": 79.0, "Macro-F1 (%)": 63.0, "Acurácia Balanceada (%)": 78.7, "Custo de API": "Zero (Local)"},
        {"Modelo Avaliado": "LLM Zero-Shot (Qwen / Groq)", "Acurácia (%)": 81.0, "Macro-F1 (%)": 79.0, "Acurácia Balanceada (%)": 78.1, "Custo de API": "API Externa"},
    ])
    col_b1, col_b2 = st.columns([1.2, 1])
    with col_b1:
        st.dataframe(df_bench, use_container_width=True, hide_index=True)
    with col_b2:
        st.bar_chart(df_bench.set_index("Modelo Avaliado")[["Acurácia Balanceada (%)", "Macro-F1 (%)"]])


# ==============================================================================
# ABA 4: CLASSIFICADOR SEMÂNTICO AO VIVO
# ==============================================================================
with aba_clf:
    st.subheader("Classificador Semântico Explicável (Taxonomia Fechada — 8 Áreas)")
    st.write(
        "Cole o enunciado de qualquer questão de Contabilidade ou do CFC. O modelo local "
        "(TF-IDF N-gramas + Regressão Logística Balanceada, custo zero de API) classifica na taxonomia fechada "
        "e evidencia os termos que mais influenciaram a decisão."
    )

    exemplo_default = (
        "Uma sociedade empresária adquiriu uma máquina industrial por R$ 200.000,00, com vida útil estimada em "
        "10 anos e valor residual de R$ 20.000,00. Considerando o método das quotas constantes previsto na NBC TG 27, "
        "qual o valor da despesa de depreciação anual?"
    )
    texto_input = st.text_area("Enunciado da questão:", value=exemplo_default, height=140)

    if st.button("🔎 Classificar Questão com IA", type="primary"):
        if len(texto_input.strip()) < 15:
            st.warning("Insira um enunciado com pelo menos 15 caracteres.")
        else:
            pred, conf, df_probs, termos = explicar_classificacao_texto(texto_input, vec_local, clf_local)
            st.success(f"**Área Predita:** {pred} &nbsp;|&nbsp; **Confiança:** {conf * 100:.1f}%")

            c_p1, c_p2 = st.columns([1.2, 1])
            with c_p1:
                st.markdown("**Distribuição de Probabilidades (Taxonomia Fechada)**")
                st.dataframe(df_probs, use_container_width=True, hide_index=True)
            with c_p2:
                st.markdown("**Termos-Chave que Explicam a Decisão (XAI)**")
                if termos:
                    df_termos = pd.DataFrame(termos, columns=["Termo / N-grama", "Peso Positivo"])
                    df_termos["Peso Positivo"] = df_termos["Peso Positivo"].round(4)
                    st.dataframe(df_termos, use_container_width=True, hide_index=True)
                else:
                    st.caption("Nenhum n-grama com peso positivo destacado na base atual.")


# ==============================================================================
# ABA 5: LOGS & AUDITORIA
# ==============================================================================
with aba_logs:
    st.subheader("Logs Auditáveis de Interação (Entrega Técnica do Canvas)")
    st.write(
        "Todas as interações do simulado diagnóstico e das recomendações explicáveis são registradas com "
        "data/hora, gabarito oficial e justificativa pedagógica para auditoria e pesquisa futura."
    )

    if st.session_state.logs:
        df_logs_view = pd.DataFrame(st.session_state.logs)
        st.dataframe(df_logs_view, use_container_width=True, hide_index=True)

        csv_bytes = df_logs_view.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button(
            label="⬇️ Baixar Logs da Sessão (.csv)",
            data=csv_bytes,
            file_name=f"logs_iaprova_{st.session_state.sessao_id}.csv",
            mime="text/csv",
            type="primary",
        )
    else:
        st.info("Nenhum log gerado ainda nesta sessão. Interaja nas abas 1 ou 2 para popular o histórico.")