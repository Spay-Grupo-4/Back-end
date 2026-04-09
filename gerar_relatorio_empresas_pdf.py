import argparse
from datetime import date
from io import BytesIO
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


def format_brl(value) -> str:
    """Formata número em formato de real brasileiro (R$ 1.234,56)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"

    formatted = f"R$ {number:,.2f}"
    # Converte do padrão en_US (1,234.56) para pt_BR (1.234,56)
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def format_cnpj(cnpj: str) -> str:
    """Aplica máscara de CNPJ em uma string numérica."""
    if cnpj is None:
        return ""
    digits = "".join(ch for ch in str(cnpj) if ch.isdigit())[:14]
    if len(digits) != 14:
        return digits
    return f"{digits[0:2]}.{digits[2:5]}.{digits[5:8]}/{digits[8:12]}-{digits[12:14]}"


def infer_porte(patrimonio):
    """Define porte estimado com base no patrimônio líquido."""
    try:
        valor = float(patrimonio)
    except (TypeError, ValueError):
        return "Micro"

    if valor < 80000:
        return "Micro"
    if valor < 500000:
        return "Medio"
    if valor < 1000000:
        return "Medio"
    return "Grande"


def infer_elegibilidade(status: str) -> str:
    """Empresas ativas são elegíveis, demais devem ser revisadas."""
    if isinstance(status, str) and status.strip().upper() == "ATIVA":
        return "Elegivel"
    return "Revisar cadastro"


def infer_potencial(status: str, patrimonio) -> str:
    """Potencial de venda estimado (Baixo/Medio/Alto)."""
    try:
        valor = float(patrimonio)
    except (TypeError, ValueError):
        valor = 0.0

    status_upper = (status or "").strip().upper()

    if status_upper != "ATIVA":
        return "Baixo"

    if valor < 50000:
        return "Baixo"
    if valor < 500000:
        return "Medio"
    return "Alto"


def infer_faixa_volume(potencial: str) -> str:
    potencial_up = (potencial or "").strip().capitalize()
    if potencial_up == "Baixo":
        return "R$ 5 mil a R$ 30 mil/mes"
    if potencial_up == "Medio":
        return "R$ 30 mil a R$ 80 mil/mes"
    return "Acima de R$ 80 mil/mes"


def infer_canal(elegibilidade: str) -> str:
    if (elegibilidade or "").strip().capitalize() == "Elegivel":
        return "Ligacao + WhatsApp"
    return "Ligacao de qualificacao"


def completar_campos(df: pd.DataFrame) -> pd.DataFrame:
    """Cria campos derivados que possam estar faltando, usando regras lógicas."""

    # Garante que o CNPJ seja tratado como string para manter zeros à esquerda
    if "cnpj" in df.columns:
        df["cnpj"] = df["cnpj"].astype(str)

    # Copia segmento digitado no front, se existir, para o campo padronizado
    if "segmento_comercial_estimado" not in df.columns:
        if "segmentoComercial" in df.columns:
            df["segmento_comercial_estimado"] = df["segmentoComercial"].fillna("Comercio Geral")
        else:
            df["segmento_comercial_estimado"] = "Comercio Geral"

    # Elegibilidade
    if "elegibilidade_lead_maquininha" not in df.columns:
        df["elegibilidade_lead_maquininha"] = df["status"].apply(infer_elegibilidade)

    # Porte estimado
    if "porte_lead_estimado" not in df.columns:
        patrimonio_col = None
        for col in ["valor_patrimonio_liquido", "patrimonioLiquido"]:
            if col in df.columns:
                patrimonio_col = col
                break
        if patrimonio_col is not None:
            df["porte_lead_estimado"] = df[patrimonio_col].apply(infer_porte)
        else:
            df["porte_lead_estimado"] = "Micro"

    # Potencial de venda
    if "potencial_venda_maquininha" not in df.columns:
        patrimonio_col = None
        for col in ["valor_patrimonio_liquido", "patrimonioLiquido"]:
            if col in df.columns:
                patrimonio_col = col
                break
        df["potencial_venda_maquininha"] = df.apply(
            lambda row: infer_potencial(row.get("status"), row.get(patrimonio_col)),
            axis=1,
        )

    # Faixa de volume mensal estimada
    if "faixa_volume_mensal_estimada" not in df.columns:
        df["faixa_volume_mensal_estimada"] = df["potencial_venda_maquininha"].apply(infer_faixa_volume)

    # Canal de abordagem sugerido
    if "canal_abordagem_sugerido" not in df.columns:
        df["canal_abordagem_sugerido"] = df["elegibilidade_lead_maquininha"].apply(infer_canal)

    # Data de ingestão (se não existir, usa a data de hoje)
    if "data_ingestao" not in df.columns:
        hoje = date.today().isoformat()
        df["data_ingestao"] = hoje

    return df


def _render_pdf(df: pd.DataFrame, pdf_canvas: canvas.Canvas) -> None:
    """Desenha o conteudo do PDF em um canvas existente."""
    width, height = A4

    margin_x = 40
    top_y = height - 40
    y = top_y

    for _, row in df.iterrows():
        # Quebra de página simples
        if y < 120:
            pdf_canvas.showPage()
            y = top_y

        pdf_canvas.setFont("Helvetica-Bold", 11)
        nome_comercial = str(row.get("nome_comercial") or row.get("nomeComercial") or "").strip()
        cnpj = format_cnpj(row.get("cnpj"))
        pdf_canvas.drawString(margin_x, y, f"{nome_comercial}  -  CNPJ: {cnpj}")
        y -= 16

        pdf_canvas.setFont("Helvetica", 10)
        nome_social = str(row.get("nome_social") or row.get("nomeSocial") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"Razao social: {nome_social}")
        y -= 14

        status = str(row.get("status") or "").strip()
        segmento = str(row.get("segmento_comercial_estimado") or "").strip()
        porte = str(row.get("porte_lead_estimado") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"Status: {status}  |  Segmento: {segmento}  |  Porte: {porte}")
        y -= 14

        uf = str(row.get("uf") or "").strip()
        municipio = str(row.get("municipio") or "").strip()
        bairro = str(row.get("bairro") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"UF: {uf}  |  Municipio: {municipio}  |  Bairro: {bairro}")
        y -= 14

        logradouro = str(row.get("logradouro") or "").strip()
        cep = str(row.get("cep") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"Logradouro: {logradouro}  |  CEP: {cep}")
        y -= 14

        telefone = str(row.get("telefone") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"Telefone: {telefone}")
        y -= 14

        patrimonio_col = "valor_patrimonio_liquido" if "valor_patrimonio_liquido" in df.columns else "patrimonioLiquido"
        patrimonio_val = row.get(patrimonio_col) if patrimonio_col in row else None
        patrimonio_fmt = format_brl(patrimonio_val)
        potencial = str(row.get("potencial_venda_maquininha") or "").strip()
        faixa = str(row.get("faixa_volume_mensal_estimada") or "").strip()
        pdf_canvas.drawString(
            margin_x,
            y,
            f"Patrimonio liquido: {patrimonio_fmt}  |  Potencial: {potencial}  |  Faixa volume: {faixa}",
        )
        y -= 14

        canal = str(row.get("canal_abordagem_sugerido") or "").strip()
        data_ingestao = str(row.get("data_ingestao") or "").strip()
        pdf_canvas.drawString(margin_x, y, f"Canal sugerido: {canal}  |  Data ingestao: {data_ingestao}")
        y -= 22

    pdf_canvas.save()


def gerar_pdf(df: pd.DataFrame, output_path: Path) -> None:
    """Gera um PDF com os dados das empresas organizados e salva em disco."""
    pdf_canvas = canvas.Canvas(str(output_path), pagesize=A4)
    _render_pdf(df, pdf_canvas)


def gerar_pdf_bytes(df: pd.DataFrame) -> bytes:
    """Gera um PDF em memoria e retorna os bytes."""
    buffer = BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=A4)
    _render_pdf(df, pdf_canvas)
    buffer.seek(0)
    return buffer.read()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ler um arquivo CSV de empresas, completar campos derivados "
            "(porte, elegibilidade, potencial, etc.) e gerar um PDF organizado."
        )
    )
    parser.add_argument(
        "--input",
        "-i",
        default="dados-empresas-projeto.csv",
        help="Caminho do arquivo CSV de entrada (default: dados-empresas-projeto.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="relatorio_empresas.pdf",
        help="Caminho do arquivo PDF de saída (default: relatorio_empresas.pdf)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Arquivo de entrada nao encontrado: {input_path}")

    # Lê CSV preservando CNPJ como texto
    df = pd.read_csv(input_path, dtype={"cnpj": str})

    df_completo = completar_campos(df.copy())

    # Gera PDF
    gerar_pdf(df_completo, output_path)

    print(f"PDF gerado em: {output_path.resolve()}")


if __name__ == "__main__":
    main()
