import os
from io import BytesIO
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from gerar_relatorio_empresas_pdf import completar_campos, gerar_pdf_bytes

try:
    import boto3
except Exception:  # noqa: BLE001
    boto3 = None


app = Flask(__name__)
CORS(app)


BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "grupo4-bronze")
BRONZE_PREFIX = os.getenv("BRONZE_PREFIX", "relatorios_formulario/")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:050665165345:envio-email")

# Parquet principal da bronze que sera atualizado a partir dos CSVs do front
CSV_MERGE_BUCKET = os.getenv("CSV_MERGE_BUCKET", BRONZE_BUCKET)
CSV_MERGE_KEY = os.getenv(
    "CSV_MERGE_KEY",
    "dados-empresas-projeto.parquet",  # arquivo existente na raiz do bucket bronze
)


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


def upload_pdf_to_bronze(pdf_bytes: bytes, object_name: str) -> str:
    """Envia o PDF gerado para o bucket S3 da bronze.

    Usa as variaveis de ambiente BRONZE_BUCKET e BRONZE_PREFIX,
    seguindo o mesmo padrao de configuracao do pipeline.
    """

    if not BRONZE_BUCKET:
        raise RuntimeError("Variavel de ambiente BRONZE_BUCKET nao definida para upload ao S3")

    key_prefix = BRONZE_PREFIX.rstrip("/")
    key = f"{key_prefix}/{object_name}".lstrip("/") if key_prefix else object_name

    s3 = get_s3_client()
    s3.put_object(Bucket=BRONZE_BUCKET, Key=key, Body=pdf_bytes, ContentType="application/pdf")
    return f"s3://{BRONZE_BUCKET}/{key}"


def merge_csv_with_parquet_s3(file_storage) -> dict:
    """Recebe um arquivo CSV (enviado pelo front), junta com o Parquet existente no S3

    e sobrescreve o Parquet unificado no bucket, removendo registros antigos com o mesmo CNPJ.
    """

    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes S3 na API")

    if not CSV_MERGE_BUCKET or not CSV_MERGE_KEY:
        raise RuntimeError("CSV_MERGE_BUCKET ou CSV_MERGE_KEY nao definidos para merge CSV+Parquet")

    # Lê CSV enviado (em memoria)
    raw = file_storage.read()
    if not raw:
        raise ValueError("Arquivo CSV vazio")

    df_novo = pd.read_csv(BytesIO(raw), dtype=str, keep_default_na=False, na_values=[])

    # Garante coluna de chave para deduplicar (CNPJ)
    if "cnpj" not in df_novo.columns:
        raise ValueError("CSV nao possui coluna 'cnpj' para usar como chave de merge")

    df_novo["cnpj"] = df_novo["cnpj"].astype(str)

    s3 = get_s3_client()

    # Tenta ler Parquet antigo do S3 (se existir)
    try:
        resp = s3.get_object(Bucket=CSV_MERGE_BUCKET, Key=CSV_MERGE_KEY)
        antigo_bytes = resp["Body"].read()
        df_antigo = pd.read_parquet(BytesIO(antigo_bytes))
        if "cnpj" in df_antigo.columns:
            df_antigo["cnpj"] = df_antigo["cnpj"].astype(str)
    except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        df_antigo = pd.DataFrame()

    # Concatena antigo + novo e remove duplicados por CNPJ, mantendo o mais recente (do CSV novo)
    if not df_antigo.empty:
        combinado = pd.concat([df_antigo, df_novo], ignore_index=True)
    else:
        combinado = df_novo.copy()

    combinado = combinado.drop_duplicates(subset=["cnpj"], keep="last").reset_index(drop=True)

    # Salva de volta em Parquet no S3
    buffer = BytesIO()
    combinado.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3.put_object(Bucket=CSV_MERGE_BUCKET, Key=CSV_MERGE_KEY, Body=buffer.read())

    return {
        "bucket": CSV_MERGE_BUCKET,
        "key": CSV_MERGE_KEY,
        "linhas_novas": len(df_novo),
        "linhas_antigas": len(df_antigo),
        "linhas_finais": len(combinado),
    }


@app.post("/leads")
def criar_lead():
    """Recebe um lead do front-end, gera PDF e envia ao S3."""
    data = request.get_json(silent=True) or {}

    required_fields = [
        "cnpj",
        "type",
        "status",
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
        "cnpj": str(data.get("cnpj", "")),
        "type": data.get("type", ""),
        "nome_social": data.get("nomeSocial", ""),
        "nome_comercial": data.get("nomeComercial", ""),
        "status": data.get("status", ""),
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

    # Cria DataFrame apenas com o lead atual (sem salvar CSV em disco)
    df = pd.DataFrame([row])

    df_completo = completar_campos(df.copy())

    try:
        # Gera PDF em memoria e envia para o S3 da bronze
        pdf_bytes = gerar_pdf_bytes(df_completo)
        pdf_s3_path = upload_pdf_to_bronze(pdf_bytes, "relatorio_empresas_formulario.pdf")
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao gerar ou enviar PDF para S3: %s", exc)
        return (
            jsonify({
                "success": False,
                "message": f"Erro ao gerar ou enviar o PDF para o S3: {exc}",
            }),
            500,
        )

    notify_success_to_sns(
        subject="[API Leads] PDF enviado com sucesso para S3",
        message=(
            "Upload de PDF concluido com sucesso via endpoint /leads.\n\n"
            f"Bucket: {BRONZE_BUCKET}\n"
            f"Arquivo: {pdf_s3_path}\n"
            f"CNPJ: {row['cnpj']}"
        ),
    )

    return jsonify({
        "success": True,
        "message": "Lead salvo e relatorio enviado para o S3.",
        "pdf_s3_path": pdf_s3_path,
    })


@app.post("/upload_csv")
def upload_csv():
    """Recebe um CSV do front, junta com o Parquet existente no S3 e sobrescreve o arquivo."""

    if "file" not in request.files:
        return (
            jsonify({"success": False, "message": "Nenhum arquivo CSV enviado (campo 'file' ausente)."}),
            400,
        )

    file_storage = request.files["file"]
    if not file_storage.filename:
        return (
            jsonify({"success": False, "message": "Nome de arquivo CSV invalido."}),
            400,
        )

    try:
        resultado = merge_csv_with_parquet_s3(file_storage)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao processar CSV e atualizar Parquet no S3: %s", exc)
        return (
            jsonify({
                "success": False,
                "message": f"Erro ao processar CSV e atualizar Parquet no S3: {exc}",
            }),
            500,
        )

    notify_success_to_sns(
        subject="[API Leads] CSV enviado com sucesso para S3",
        message=(
            "Upload de CSV concluido com sucesso via endpoint /upload_csv.\n\n"
            f"Arquivo recebido: {file_storage.filename}\n"
            f"Bucket destino: {resultado['bucket']}\n"
            f"Chave destino: {resultado['key']}\n"
            f"Linhas novas: {resultado['linhas_novas']}\n"
            f"Linhas antigas: {resultado['linhas_antigas']}\n"
            f"Linhas finais: {resultado['linhas_finais']}"
        ),
    )

    return jsonify({
        "success": True,
        "message": "CSV processado e Parquet atualizado no S3.",
        "details": resultado,
    })


if __name__ == "__main__":
    # Exemplo de execucao: python api_leads.py
    app.run(host="0.0.0.0", port=5000, debug=True)
