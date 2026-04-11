from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from io import BytesIO
from typing import Any

import boto3
import pandas as pd

SUPPORTED_EXTENSIONS = (".parquet", ".csv")
DEFAULT_SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:050665165345:envio-email"


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_s3_client():
    return boto3.client("s3")


def get_sns_client():
    return boto3.client("sns")


def notify_sns(topic_arn: str, subject: str, message: str, logger: logging.Logger) -> tuple[bool, str]:
    if not topic_arn:
        logger.warning("SNS_TOPIC_ARN nao informado; notificacao SNS nao sera enviada.")
        return False, "SNS_TOPIC_ARN vazio"

    try:
        response = get_sns_client().publish(TopicArn=topic_arn, Subject=subject, Message=message)
        message_id = response.get("MessageId", "")
        logger.info("Notificacao SNS enviada para %s (MessageId=%s)", topic_arn, message_id)
        return True, message_id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Falha ao enviar notificacao SNS: %s", exc)
        return False, str(exc)


def detect_encoding_from_bytes(raw: bytes, default: str = "utf-8") -> str:
    for enc in (default, "latin-1"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return default


def detect_csv_separator_from_text(sample: str) -> str:
    if sample.count(";") > sample.count(","):
        return ";"
    return ","


def detect_csv_separator_from_bytes(raw: bytes, encoding: str) -> str:
    sample = raw[:5000].decode(encoding, errors="replace")
    return detect_csv_separator_from_text(sample)


def list_bronze_objects(bronze_bucket: str, bronze_prefix: str, logger: logging.Logger) -> list[dict]:
    s3 = get_s3_client()
    prefix = bronze_prefix or ""
    objects: list[dict] = []

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bronze_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj.get("Key", "")
            if not key or key.endswith("/"):
                continue
            if not key.lower().endswith(SUPPORTED_EXTENSIONS):
                continue
            objects.append(obj)

    logger.info("Objetos elegiveis encontrados na bronze: %s", len(objects))
    return objects


def resolve_bronze_key(bronze_bucket: str, bronze_key: str, bronze_prefix: str, logger: logging.Logger) -> str:
    if bronze_key:
        return bronze_key

    objects = list_bronze_objects(bronze_bucket, bronze_prefix, logger)
    if not objects:
        raise FileNotFoundError(
            f"Nenhum objeto elegivel ({', '.join(SUPPORTED_EXTENSIONS)}) encontrado em s3://{bronze_bucket}/{bronze_prefix}"
        )

    prioritized = sorted(
        objects,
        key=lambda item: (
            not str(item.get("Key", "")).lower().endswith(".parquet"),
            "empresa" not in str(item.get("Key", "")).lower(),
            -float(item.get("LastModified").timestamp()) if item.get("LastModified") else 0,
        ),
    )
    selected_key = str(prioritized[0]["Key"])
    logger.info("BRONZE_KEY nao informada. Objeto selecionado automaticamente: %s", selected_key)
    return selected_key


def resolve_optional_bronze_key_by_term(
    bronze_bucket: str,
    bronze_prefix: str,
    search_term: str,
    logger: logging.Logger,
) -> str | None:
    """Busca opcional de arquivo na bronze pelo termo informado no nome da chave."""

    objects = list_bronze_objects(bronze_bucket, bronze_prefix, logger)
    term = (search_term or "").strip().lower()
    if not term:
        return None

    candidates = [
        obj
        for obj in objects
        if term in str(obj.get("Key", "")).lower()
    ]

    if not candidates:
        logger.info("Nenhum arquivo com termo '%s' foi encontrado na bronze.", term)
        return None

    candidates = sorted(
        candidates,
        key=lambda item: -float(item.get("LastModified").timestamp()) if item.get("LastModified") else 0,
    )
    selected_key = str(candidates[0]["Key"])
    logger.info("Arquivo opcional selecionado automaticamente para '%s': %s", term, selected_key)
    return selected_key


def read_bronze_from_s3(
    bronze_bucket: str,
    bronze_key: str,
    bronze_prefix: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    bronze_key = resolve_bronze_key(bronze_bucket, bronze_key, bronze_prefix, logger)
    return read_bronze_key_from_s3(bronze_bucket, bronze_key, logger)


def read_bronze_key_from_s3(
    bronze_bucket: str,
    bronze_key: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Lendo bronze no S3: s3://%s/%s", bronze_bucket, bronze_key)
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bronze_bucket, Key=bronze_key)
    raw = response["Body"].read()

    suffix = os.path.splitext(bronze_key)[1].lower()
    if suffix == ".parquet":
        return pd.read_parquet(BytesIO(raw))

    encoding = detect_encoding_from_bytes(raw)
    separator = detect_csv_separator_from_bytes(raw, encoding)

    logger.info("Encoding detectado no S3: %s", encoding)
    logger.info("Separador detectado no S3: %s", separator)

    return pd.read_csv(
        BytesIO(raw),
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding=encoding,
        sep=separator,
        low_memory=False,
    )


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_column_name(name: str) -> str:
    text = normalize_text(name).upper()
    text = re.sub(r"[^A-Z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def clean_empresas_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df.empty:
        logger.warning("Arquivo de entrada sem linhas. Nada para tratar.")
        return df

    renamed_columns = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=renamed_columns)

    for col in df.columns:
        df[col] = df[col].apply(lambda value: "" if pd.isna(value) else normalize_text(value).upper())

    before_rows = len(df)
    df = df.replace("", pd.NA)
    df = df.drop_duplicates().copy()
    df = df.dropna(how="all").copy()
    df = df.fillna("")
    after_rows = len(df)

    logger.info("Registros antes da limpeza: %s", before_rows)
    logger.info("Registros depois da limpeza: %s", after_rows)
    return df


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def upload_bytes_to_s3(data: bytes, bucket: str, key: str, logger: logging.Logger) -> None:
    get_s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    logger.info("Enviado para S3: s3://%s/%s", bucket, key)


def salvar_dataset_no_s3(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
    file_name: str,
    logger: logging.Logger,
) -> dict[str, str]:
    prefix = prefix.rstrip("/")
    parquet_key = f"{prefix}/{file_name}" if prefix else file_name
    upload_bytes_to_s3(dataframe_to_parquet_bytes(df), bucket, parquet_key, logger)
    return {"parquet": parquet_key}


def salvar_silver_no_s3(df: pd.DataFrame, bucket: str, prefix: str, logger: logging.Logger) -> dict[str, str]:
    return salvar_dataset_no_s3(df, bucket, prefix, "empresas_tratadas.parquet", logger)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ANN401
    setup_logging()
    logger = get_logger("lambda_bronze_to_silver")

    payload = event or {}
    sns_topic_arn = DEFAULT_SNS_TOPIC_ARN

    try:
        bronze_bucket = payload.get("bronze_bucket") or os.getenv("BRONZE_BUCKET", "grupo4-bronze")
        bronze_key = payload.get("bronze_key") or os.getenv("BRONZE_KEY", "")
        bronze_prefix = payload.get("bronze_prefix") or os.getenv("BRONZE_PREFIX", "")
        crm_bronze_key = payload.get("crm_bronze_key") or os.getenv("CRM_BRONZE_KEY", "")

        silver_bucket = payload.get("silver_bucket") or os.getenv("SILVER_BUCKET", "grupo4-silver")
        silver_prefix = payload.get("silver_prefix") or os.getenv("SILVER_PREFIX", "empresas/")
        crm_silver_prefix = payload.get("crm_silver_prefix") or os.getenv("CRM_SILVER_PREFIX", "crm_vendas/")

        empresas_bronze = read_bronze_from_s3(bronze_bucket, bronze_key, bronze_prefix, logger)
        total_empresas_bronze = len(empresas_bronze)

        empresas_silver = clean_empresas_dataframe(empresas_bronze, logger)
        total_empresas_silver = len(empresas_silver)
        linhas_removidas = total_empresas_bronze - total_empresas_silver

        silver_paths = salvar_silver_no_s3(empresas_silver, silver_bucket, silver_prefix, logger)

        crm_dataset = {
            "processed": False,
            "bronze_key": "",
            "silver_paths": {},
            "total_registros_bronze": 0,
            "total_registros_silver": 0,
            "linhas_removidas_durante_limpeza": 0,
        }

        if not crm_bronze_key:
            crm_bronze_key = resolve_optional_bronze_key_by_term(
                bronze_bucket,
                bronze_prefix,
                "base_interna_crm_vendas",
                logger,
            ) or ""

        if crm_bronze_key:
            crm_bronze = read_bronze_key_from_s3(bronze_bucket, crm_bronze_key, logger)
            total_crm_bronze = len(crm_bronze)

            crm_silver = clean_empresas_dataframe(crm_bronze, logger)
            total_crm_silver = len(crm_silver)
            crm_linhas_removidas = total_crm_bronze - total_crm_silver

            crm_silver_paths = salvar_dataset_no_s3(
                crm_silver,
                silver_bucket,
                crm_silver_prefix,
                "base_interna_crm_vendas_tratada.parquet",
                logger,
            )

            crm_dataset = {
                "processed": True,
                "bronze_key": crm_bronze_key,
                "silver_paths": crm_silver_paths,
                "total_registros_bronze": total_crm_bronze,
                "total_registros_silver": total_crm_silver,
                "linhas_removidas_durante_limpeza": crm_linhas_removidas,
            }

        body = {
            "message": "Pipeline bronze->silver executado com sucesso.",
            "bronze_bucket": bronze_bucket,
            "silver_bucket": silver_bucket,
            "silver_paths": silver_paths,
            "crm_dataset": crm_dataset,
            "total_empresas_bronze": total_empresas_bronze,
            "total_empresas_silver": total_empresas_silver,
            "linhas_removidas_durante_limpeza": linhas_removidas,
        }

        sns_sent, sns_info = notify_sns(
            sns_topic_arn,
            "[Notificacao Lambda] Sucesso bronze->silver",
            (
                "Fluxo bronze->silver executado com sucesso.\n\n"
                f"Bucket bronze: {bronze_bucket}\n"
                f"Bucket silver: {silver_bucket}\n"
                f"Registros bronze: {total_empresas_bronze}\n"
                f"Registros silver: {total_empresas_silver}\n"
                f"Linhas removidas: {linhas_removidas}\n"
                f"Arquivos silver: {json.dumps(silver_paths, ensure_ascii=False)}\n\n"
                "Processamento adicional CRM:\n"
                f"- Processado: {crm_dataset['processed']}\n"
                f"- Bronze key CRM: {crm_dataset['bronze_key']}\n"
                f"- Arquivos silver CRM: {json.dumps(crm_dataset['silver_paths'], ensure_ascii=False)}"
            ),
            logger,
        )

        body["sns"] = {
            "topic_arn": sns_topic_arn,
            "sent": sns_sent,
            "info": sns_info,
        }

        return {"statusCode": 200, "body": json.dumps(body, ensure_ascii=False)}

    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro no fluxo bronze->silver: %s", exc)

        sns_sent, sns_info = notify_sns(
            sns_topic_arn,
            "[Notificacao Lambda] Erro bronze->silver",
            f"Falha no fluxo bronze->silver.\n\nErro: {exc}",
            logger,
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Erro no fluxo bronze->silver.",
                    "error": str(exc),
                    "sns": {
                        "topic_arn": sns_topic_arn,
                        "sent": sns_sent,
                        "info": sns_info,
                    },
                },
                ensure_ascii=False,
            ),
        }
