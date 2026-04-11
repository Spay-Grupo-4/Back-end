import os
import random
import re
import unicodedata
from datetime import date
from io import BytesIO

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

try:
    import boto3
except Exception:  # noqa: BLE001
    boto3 = None


app = Flask(__name__)
CORS(app)


BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "grupo4-bronze")
BRONZE_PREFIX = os.getenv("BRONZE_PREFIX", "relatorios_formulario/")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:050665165345:envio-email")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "grupo4-silver")
SILVER_PREFIX = os.getenv("SILVER_PREFIX", "empresas/")
SILVER_KEY = os.getenv("SILVER_KEY", f"{SILVER_PREFIX.rstrip('/')}/empresas_tratadas.parquet")


def get_s3_client():
    """Retorna um client S3 usando boto3 apenas para a API.

    Nao depende do pipeline, para evitar importar outras bibliotecas.
    """

    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes S3 na API")

    return boto3.client("s3")


def get_sns_client():
    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes SNS na API")

    return boto3.client("sns")


def notify_success_to_sns(subject: str, message: str) -> None:
    """Publica notificacao no topico SNS quando houver sucesso em upload."""

    if not SNS_TOPIC_ARN:
        app.logger.info("SNS_TOPIC_ARN nao configurado. Notificacao ignorada.")
        return

    try:
        sns = get_sns_client()
        sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=message)
        app.logger.info("Notificacao SNS enviada para %s", SNS_TOPIC_ARN)
    except Exception as exc:  # noqa: BLE001
        # Falha de notificacao nao deve quebrar a API.
        app.logger.exception("Falha ao enviar notificacao SNS: %s", exc)


def parse_brl_to_float(value):
    """Converte string em formato monetario BRL para float simples."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.replace("R$", "").replace(" ", "")
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def upload_pdf_to_bronze(pdf_bytes: bytes, object_name: str, metadata: dict | None = None) -> str:
    """Envia um PDF para o bucket S3 da bronze.

    Usa as variaveis de ambiente BRONZE_BUCKET e BRONZE_PREFIX,
    seguindo o mesmo padrao de configuracao do pipeline.
    """

    if not BRONZE_BUCKET:
        raise RuntimeError("Variavel de ambiente BRONZE_BUCKET nao definida para upload ao S3")

    key_prefix = BRONZE_PREFIX.rstrip("/")
    key = f"{key_prefix}/{object_name}".lstrip("/") if key_prefix else object_name

    s3 = get_s3_client()
    kwargs = {
        "Bucket": BRONZE_BUCKET,
        "Key": key,
        "Body": pdf_bytes,
        "ContentType": "application/pdf",
    }
    if metadata:
        kwargs["Metadata"] = metadata

    s3.put_object(**kwargs)
    return f"s3://{BRONZE_BUCKET}/{key}"


def normalize_text(value):
    if value is None:
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_column_name(name: str) -> str:
    text = normalize_text(name)
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def only_digits(value) -> str:
    return re.sub(r"\D+", "", str(value or ""))


def clean_empresas_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Replica tratamento da silver da lambda (normalizacao de colunas e valores)."""
    if df.empty:
        return df

    renamed_columns = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=renamed_columns)

    for col in df.columns:
        df[col] = df[col].apply(lambda value: "" if pd.isna(value) else normalize_text(value).upper())

    # Mantem CNPJ somente com numeros para padronizacao e deduplicacao correta.
    if "CNPJ" in df.columns:
        df["CNPJ"] = df["CNPJ"].apply(only_digits)

    df = df.replace("", pd.NA)
    df = df.drop_duplicates().copy()
    df = df.dropna(how="all").copy()
    df = df.fillna("")

    return df


def choose_weighted(options: list[str], weights: list[int]) -> str:
    return random.choices(options, weights=weights, k=1)[0]


def infer_porte_estimado(patrimonio: float | None) -> str:
    if patrimonio is None:
        return choose_weighted(["Micro", "Medio"], [70, 30])

    if patrimonio < 80000:
        return choose_weighted(["Micro", "Medio"], [85, 15])
    if patrimonio < 1000000:
        return choose_weighted(["Medio", "Micro", "Grande"], [75, 20, 5])
    return choose_weighted(["Grande", "Medio"], [85, 15])


def infer_potencial_venda(porte: str, segmento: str) -> str:
    porte_up = (porte or "").strip().upper()
    segmento_up = (segmento or "").strip().upper()

    if porte_up == "MICRO":
        potencial = choose_weighted(["Baixo", "Medio", "Alto"], [70, 25, 5])
    elif porte_up == "MEDIO":
        potencial = choose_weighted(["Baixo", "Medio", "Alto"], [25, 55, 20])
    else:
        potencial = choose_weighted(["Baixo", "Medio", "Alto"], [10, 40, 50])

    segmentos_quentes = ["E-COMMERCE", "SUPERMERC", "ATACAD", "RESTAUR", "POSTO", "DISTRIB"]
    if any(token in segmento_up for token in segmentos_quentes):
        if potencial == "Baixo" and random.random() < 0.6:
            return "Medio"
        if potencial == "Medio" and random.random() < 0.35:
            return "Alto"

    return potencial


def infer_faixa_volume(potencial: str) -> str:
    potencial_up = (potencial or "").strip().upper()
    if potencial_up == "BAIXO":
        return choose_weighted(
            [
                "R$ 5 mil a R$ 30 mil/mes",
                "R$ 8 mil a R$ 25 mil/mes",
            ],
            [75, 25],
        )
    if potencial_up == "MEDIO":
        return choose_weighted(
            [
                "R$ 30 mil a R$ 80 mil/mes",
                "R$ 35 mil a R$ 90 mil/mes",
            ],
            [75, 25],
        )
    return choose_weighted(
        [
            "Acima de R$ 80 mil/mes",
            "Acima de R$ 100 mil/mes",
        ],
        [75, 25],
    )


def infer_canal_abordagem(elegibilidade: str, potencial: str) -> str:
    eleg_up = (elegibilidade or "").strip().upper()
    potencial_up = (potencial or "").strip().upper()

    if eleg_up != "ELEGIVEL":
        return "Ligacao de qualificacao"

    if potencial_up == "ALTO":
        return choose_weighted(["Visita consultiva + ligacao", "Ligacao + WhatsApp"], [60, 40])
    if potencial_up == "MEDIO":
        return choose_weighted(["Ligacao + WhatsApp", "WhatsApp"], [75, 25])
    return choose_weighted(["WhatsApp", "Ligacao + WhatsApp"], [70, 30])


def build_auto_lead_fields(row: dict, patrimonio: float | None) -> dict:
    """Gera campos automaticos para cadastro manual com logica aleatoria coerente."""

    status = (row.get("status") or "").strip().upper()
    segmento = row.get("segmento_comercial_estimado", "")

    elegibilidade = "Elegivel" if status == "ATIVA" else "Revisar cadastro"
    porte = infer_porte_estimado(patrimonio)
    potencial = infer_potencial_venda(porte, str(segmento))
    faixa = infer_faixa_volume(potencial)
    canal = infer_canal_abordagem(elegibilidade, potencial)

    return {
        "elegibilidade_lead_maquininha": elegibilidade,
        "porte_lead_estimado": porte,
        "potencial_venda_maquininha": potencial,
        "faixa_volume_mensal_estimada": faixa,
        "canal_abordagem_sugerido": canal,
        "data_ingestao": date.today().isoformat(),
    }


def merge_lead_with_silver_parquet_s3(df_novo: pd.DataFrame) -> dict:
    """Recebe um DF de lead manual, trata como silver e faz merge por CNPJ no parquet silver.

    Sobrescreve o parquet mantendo o registro mais recente por CNPJ.
    """

    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes S3 na API")

    if not SILVER_BUCKET or not SILVER_KEY:
        raise RuntimeError("SILVER_BUCKET ou SILVER_KEY nao definidos para merge na silver")

    df_novo = clean_empresas_dataframe(df_novo.copy())

    if "CNPJ" not in df_novo.columns:
        raise ValueError("Dados nao possuem coluna 'CNPJ' para usar como chave de merge")

    df_novo["CNPJ"] = df_novo["CNPJ"].astype(str)

    s3 = get_s3_client()

    # Tenta ler Parquet antigo da silver (se existir)
    try:
        resp = s3.get_object(Bucket=SILVER_BUCKET, Key=SILVER_KEY)
        antigo_bytes = resp["Body"].read()
        df_antigo = pd.read_parquet(BytesIO(antigo_bytes))
        if "CNPJ" in df_antigo.columns:
            df_antigo["CNPJ"] = df_antigo["CNPJ"].astype(str)
    except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        df_antigo = pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            df_antigo = pd.DataFrame()
        else:
            raise

    # Concatena antigo + novo e remove duplicados por CNPJ, mantendo o mais recente
    if not df_antigo.empty:
        combinado = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        combinado = df_novo.copy()

    combinado = clean_empresas_dataframe(combinado)
    combinado = combinado.drop_duplicates(subset=["CNPJ"], keep="last").reset_index(drop=True)

    # Salva de volta em Parquet na silver
    buffer = BytesIO()
    combinado.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3.put_object(Bucket=SILVER_BUCKET, Key=SILVER_KEY, Body=buffer.read())

    return {
        "bucket": SILVER_BUCKET,
        "key": SILVER_KEY,
        "linhas_novas": len(df_novo),
        "linhas_antigas": len(df_antigo),
        "linhas_finais": len(combinado),
    }


@app.post("/leads")
def criar_lead():
    """Recebe um lead do front-end e salva direto na silver (parquet tratado)."""
    data = request.get_json(silent=True) or {}

    required_fields = [
        "cnpj",
        "type",
        "nomeSocial",
        "nomeComercial",
    ]
    missing = [f for f in required_fields if not data.get(f)]
    if missing:
        return (
            jsonify({
                "success": False,
                "message": f"Campos obrigatorios ausentes: {', '.join(missing)}",
            }),
            400,
        )

    patrimonio_float = parse_brl_to_float(data.get("patrimonioLiquido"))

    # Mapeia campos vindos do front para o esquema do CSV/projeto
    row = {
        "cnpj": only_digits(data.get("cnpj", "")),
        "type": data.get("type", ""),
        "nome_social": data.get("nomeSocial", ""),
        "nome_comercial": data.get("nomeComercial", ""),
        "status": "ATIVA",
        "telefone": data.get("telefone", ""),
        "cep": data.get("cep", ""),
        "uf": data.get("uf", ""),
        "municipio": data.get("municipio", ""),
        "bairro": data.get("bairro", ""),
        "complemento": data.get("complemento", ""),
        "logradouro": data.get("logradouro", ""),
        "valor_patrimonio_liquido": patrimonio_float,
        "data_inicio_situacao": data.get("dataInicioSituacao", ""),
        "data_registro": data.get("dataRegistro", ""),
        "segmento_comercial_estimado": data.get("segmentoComercial", ""),
    }

    row.update(build_auto_lead_fields(row, patrimonio_float))

    df = pd.DataFrame([row])

    try:
        resultado = merge_lead_with_silver_parquet_s3(df)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao tratar e salvar lead na silver: %s", exc)
        return (
            jsonify({
                "success": False,
                "message": f"Erro ao tratar e salvar lead na silver: {exc}",
            }),
            500,
        )

    notify_success_to_sns(
        subject="[API Leads] Lead manual salvo com sucesso na silver",
        message=(
            "Cadastro manual concluido com sucesso via endpoint /leads.\n\n"
            f"Bucket silver: {resultado['bucket']}\n"
            f"Chave silver: {resultado['key']}\n"
            f"Linhas novas: {resultado['linhas_novas']}\n"
            f"Linhas antigas: {resultado['linhas_antigas']}\n"
            f"Linhas finais: {resultado['linhas_finais']}\n"
            f"CNPJ: {row['cnpj']}"
        ),
    )

    return jsonify({
        "success": True,
        "message": "Lead tratado e salvo com sucesso na silver.",
        "details": resultado,
    })


@app.post("/upload_pdf")
def upload_pdf():
    """Recebe um PDF enviado manualmente e envia para a bronze."""

    if "file" not in request.files:
        return (
            jsonify({"success": False, "message": "Nenhum arquivo PDF enviado (campo 'file' ausente)."}),
            400,
        )

    file_storage = request.files["file"]
    if not file_storage.filename:
        return (
            jsonify({"success": False, "message": "Nome de arquivo PDF invalido."}),
            400,
        )

    filename = file_storage.filename
    if not filename.lower().endswith(".pdf"):
        return (
            jsonify({"success": False, "message": "Formato invalido. Envie um arquivo .pdf"}),
            400,
        )

    try:
        pdf_bytes = file_storage.read()
        if not pdf_bytes:
            raise ValueError("Arquivo PDF vazio")

        data_ingestao = date.today().isoformat()
        pdf_s3_path = upload_pdf_to_bronze(
            pdf_bytes,
            filename,
            metadata={"data_ingestao": data_ingestao},
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao enviar PDF para bronze no S3: %s", exc)
        return (
            jsonify({
                "success": False,
                "message": f"Erro ao enviar PDF para bronze no S3: {exc}",
            }),
            500,
        )

    notify_success_to_sns(
        subject="[API Leads] PDF enviado com sucesso para bronze",
        message=(
            "Upload de PDF concluido com sucesso via endpoint /upload_pdf.\n\n"
            f"Arquivo recebido: {filename}\n"
            f"Destino: {pdf_s3_path}\n"
            f"Data ingestao: {data_ingestao}"
        ),
    )

    return jsonify({
        "success": True,
        "message": "PDF enviado com sucesso para bronze.",
        "pdf_s3_path": pdf_s3_path,
        "data_ingestao": data_ingestao,
    })


if __name__ == "__main__":
    # Exemplo de execucao: python api_leads.py
    app.run(host="0.0.0.0", port=5000, debug=True)
