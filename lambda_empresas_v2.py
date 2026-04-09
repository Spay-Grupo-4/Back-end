from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from io import BytesIO
from typing import Any, Dict, List, Tuple

import boto3
import pandas as pd


# ==========================
# Utilidades gerais
# ==========================

SUPPORTED_EXTENSIONS = (".parquet", ".csv")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ==========================
# Leitura da bronze (S3)
# ==========================


def get_s3_client():
    return boto3.client("s3")


def get_sns_client():
    return boto3.client("sns")


def detect_encoding_from_bytes(raw: bytes, default: str = "utf-8") -> str:
    """Tenta detectar um encoding simples sem depender de bibliotecas extras.

    Primeiro tenta decodificar como UTF-8; se falhar, tenta latin-1.
    Caso nenhuma funcione, retorna o default informado.
    """

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


def list_bronze_objects(bronze_bucket: str, bronze_prefix: str, logger: logging.Logger) -> List[dict]:
    s3 = get_s3_client()
    prefix = bronze_prefix or ""
    objects: List[dict] = []

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

    # Prioriza chaves parquet com 'empresa' no nome e, depois, o arquivo mais recente
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


def read_bronze_from_s3(
    bronze_bucket: str,
    bronze_key: str,
    bronze_prefix: str,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, str]:
    if not bronze_bucket:
        raise ValueError("Informe bronze_bucket para leitura da bronze no S3")

    bronze_key = resolve_bronze_key(bronze_bucket, bronze_key, bronze_prefix, logger)

    logger.info("Lendo bronze no S3: s3://%s/%s", bronze_bucket, bronze_key)
    s3 = get_s3_client()
    response = s3.get_object(Bucket=bronze_bucket, Key=bronze_key)
    raw = response["Body"].read()

    suffix = os.path.splitext(bronze_key)[1].lower()
    if suffix == ".parquet":
        df = pd.read_parquet(BytesIO(raw))
        return df, ","

    encoding = detect_encoding_from_bytes(raw)
    separator = detect_csv_separator_from_bytes(raw, encoding)

    logger.info("Encoding detectado no S3: %s", encoding)
    logger.info("Separador detectado no S3: %s", separator)

    df = pd.read_csv(
        BytesIO(raw),
        dtype=str,
        keep_default_na=False,
        na_values=[],
        encoding=encoding,
        sep=separator,
        low_memory=False,
    )

    return df, separator


# ==========================
# Normalizacao e limpeza (silver)
# ==========================


def normalize_text(value: Any) -> str:
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


def clean_empresas_dataframe(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    if df.empty:
        logger.warning("Arquivo de entrada sem linhas. Nada para tratar.")
        return df

    # Padroniza nomes de colunas
    renamed_columns = {col: normalize_column_name(col) for col in df.columns}
    df = df.rename(columns=renamed_columns)

    # Padroniza todos os valores textuais
    for col in df.columns:
        df[col] = df[col].apply(lambda value: "" if pd.isna(value) else normalize_text(value).upper())

    # Remove duplicidades exatas e linhas totalmente vazias
    before_rows = len(df)
    df = df.replace("", pd.NA)
    df = df.drop_duplicates().copy()
    df = df.dropna(how="all").copy()
    df = df.fillna("")
    after_rows = len(df)

    logger.info("Registros antes da limpeza: %s", before_rows)
    logger.info("Registros depois da limpeza: %s", after_rows)

    return df


# ==========================
# Funcoes auxiliares de metricas (gold)
# ==========================


def to_numeric_series(series: pd.Series) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.replace(r"[^0-9,.-]", "", regex=True)
        .str.replace(".", "", regex=False)
        .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(normalized, errors="coerce")


def to_digits(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\D+", "", regex=True)


def is_valid_cnpj(series: pd.Series) -> pd.Series:
    digits = to_digits(series)
    valid_len = digits.str.len().eq(14)
    repeated_digits = digits.apply(lambda value: len(set(value)) == 1 if isinstance(value, str) and value else False)
    return valid_len & ~repeated_digits


def is_valid_phone(series: pd.Series) -> pd.Series:
    digits = to_digits(series)
    return digits.str.len().isin([10, 11])


def parse_faixa_volume_to_midpoint(series: pd.Series) -> pd.Series:
    text = series.astype(str)

    low_high = text.str.extract(r"R\$\s*([\d.,]+)\s*MIL\s*A\s*R\$\s*([\d.,]+)\s*MIL", expand=True)
    low = pd.to_numeric(low_high[0].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    high = pd.to_numeric(low_high[1].str.replace(".", "", regex=False).str.replace(",", ".", regex=False), errors="coerce")
    midpoint = ((low + high) / 2.0) * 1000

    only_above = text.str.extract(r"ACIMA\s*DE\s*R\$\s*([\d.,]+)\s*MIL", expand=True)
    above_value = pd.to_numeric(
        only_above[0].str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )
    midpoint = midpoint.fillna(above_value * 1000)

    return midpoint


def build_gold_metrics(empresas_silver: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df = empresas_silver.copy()

    total_empresas = len(df)
    status_col = "STATUS"
    eleg_col = "ELEGIBILIDADE_LEAD_MAQUININHA"
    pot_col = "POTENCIAL_VENDA_MAQUININHA"
    uf_col = "UF"
    municipio_col = "MUNICIPIO"
    segmento_col = "SEGMENTO_COMERCIAL_ESTIMADO"
    patrimonio_col = "VALOR_PATRIMONIO_LIQUIDO"
    cnpj_col = "CNPJ"
    telefone_col = "TELEFONE"
    porte_col = "PORTE_LEAD_ESTIMADO"
    faixa_volume_col = "FAIXA_VOLUME_MENSAL_ESTIMADA"
    canal_col = "CANAL_ABORDAGEM_SUGERIDO"
    data_ingestao_col = "DATA_INGESTAO"

    status_ativa = (df[status_col] == "ATIVA").sum() if status_col in df.columns else 0
    elegiveis = (df[eleg_col] == "ELEGIVEL").sum() if eleg_col in df.columns else 0
    potencial_alto = (df[pot_col] == "ALTO").sum() if pot_col in df.columns else 0
    ativas_e_elegiveis = (
        ((df[status_col] == "ATIVA") & (df[eleg_col] == "ELEGIVEL")).sum()
        if status_col in df.columns and eleg_col in df.columns
        else 0
    )

    patrimonio = pd.Series(dtype="float64")
    if patrimonio_col in df.columns:
        patrimonio = to_numeric_series(df[patrimonio_col])

    volume_midpoint = pd.Series(dtype="float64")
    if faixa_volume_col in df.columns:
        volume_midpoint = parse_faixa_volume_to_midpoint(df[faixa_volume_col])

    cnpj_valid = pd.Series(dtype="bool")
    if cnpj_col in df.columns:
        cnpj_valid = is_valid_cnpj(df[cnpj_col])

    telefone_valid = pd.Series(dtype="bool")
    if telefone_col in df.columns:
        telefone_valid = is_valid_phone(df[telefone_col])

    # Lead acionavel: ATIVA + ELEGIVEL + CNPJ valido + telefone valido
    lead_acionavel = pd.Series([False] * len(df))
    if status_col in df.columns and eleg_col in df.columns and not cnpj_valid.empty and not telefone_valid.empty:
        lead_acionavel = (df[status_col] == "ATIVA") & (df[eleg_col] == "ELEGIVEL") & cnpj_valid & telefone_valid

    conversao_elegiveis = (elegiveis / total_empresas) * 100 if total_empresas else 0
    conversao_potencial_alto_sobre_elegiveis = (potencial_alto / elegiveis) * 100 if elegiveis else 0
    conversao_acionaveis = (lead_acionavel.sum() / total_empresas) * 100 if total_empresas else 0

    data_ingestao = pd.Series(dtype="datetime64[ns]")
    if data_ingestao_col in df.columns:
        data_ingestao = pd.to_datetime(df[data_ingestao_col], errors="coerce")

    ultima_data_ingestao = ""
    primeira_data_ingestao = ""
    if not data_ingestao.empty and data_ingestao.notna().any():
        ultima_data_ingestao = data_ingestao.max().date().isoformat()
        primeira_data_ingestao = data_ingestao.min().date().isoformat()

    resumo_data = [
        {"METRICA": "TOTAL_EMPRESAS", "VALOR": total_empresas},
        {"METRICA": "STATUS_ATIVA_TOTAL", "VALOR": int(status_ativa)},
        {
            "METRICA": "STATUS_ATIVA_PERCENTUAL",
            "VALOR": round((status_ativa / total_empresas) * 100, 2) if total_empresas else 0,
        },
        {"METRICA": "ELEGIVEIS_TOTAL", "VALOR": int(elegiveis)},
        {
            "METRICA": "ELEGIVEIS_PERCENTUAL",
            "VALOR": round((elegiveis / total_empresas) * 100, 2) if total_empresas else 0,
        },
        {"METRICA": "POTENCIAL_ALTO_TOTAL", "VALOR": int(potencial_alto)},
        {
            "METRICA": "POTENCIAL_ALTO_PERCENTUAL",
            "VALOR": round((potencial_alto / total_empresas) * 100, 2) if total_empresas else 0,
        },
        {"METRICA": "PATRIMONIO_TOTAL", "VALOR": float(patrimonio.sum(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "PATRIMONIO_MEDIO", "VALOR": float(patrimonio.mean(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "PATRIMONIO_MEDIANO", "VALOR": float(patrimonio.median(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "ATIVAS_E_ELEGIVEIS_TOTAL", "VALOR": int(ativas_e_elegiveis)},
        {"METRICA": "TAXA_ELEGIVEIS", "VALOR": round(conversao_elegiveis, 2)},
        {
            "METRICA": "TAXA_POTENCIAL_ALTO_SOBRE_ELEGIVEIS",
            "VALOR": round(conversao_potencial_alto_sobre_elegiveis, 2),
        },
        {
            "METRICA": "VOLUME_POTENCIAL_MENSAL_ESTIMADO_TOTAL",
            "VALOR": float(volume_midpoint.sum(skipna=True)) if not volume_midpoint.empty else 0,
        },
        {
            "METRICA": "VOLUME_POTENCIAL_MENSAL_ESTIMADO_MEDIO",
            "VALOR": float(volume_midpoint.mean(skipna=True)) if not volume_midpoint.empty else 0,
        },
        {
            "METRICA": "CNPJ_VALIDO_PERCENTUAL",
            "VALOR": round(float(cnpj_valid.mean() * 100), 2) if not cnpj_valid.empty else 0,
        },
        {
            "METRICA": "TELEFONE_VALIDO_PERCENTUAL",
            "VALOR": round(float(telefone_valid.mean() * 100), 2) if not telefone_valid.empty else 0,
        },
        {
            "METRICA": "LEADS_ACIONAVEIS_TOTAL",
            "VALOR": int(lead_acionavel.sum()) if not lead_acionavel.empty else 0,
        },
        {
            "METRICA": "LEADS_ACIONAVEIS_PERCENTUAL",
            "VALOR": round(conversao_acionaveis, 2),
        },
        {
            "METRICA": "REGISTROS_COM_DATA_INGESTAO_TOTAL",
            "VALOR": int(data_ingestao.notna().sum()) if not data_ingestao.empty else 0,
        },
    ]
    resumo = pd.DataFrame(resumo_data)

    status = pd.DataFrame(columns=["STATUS", "TOTAL"])
    if status_col in df.columns:
        status = (
            df[status_col]
            .value_counts(dropna=False)
            .rename_axis("STATUS")
            .reset_index(name="TOTAL")
        )

    por_uf = pd.DataFrame(columns=["UF", "TOTAL"])
    if uf_col in df.columns:
        por_uf = (
            df[uf_col]
            .value_counts(dropna=False)
            .rename_axis("UF")
            .reset_index(name="TOTAL")
        )

    por_segmento = pd.DataFrame(columns=["SEGMENTO_COMERCIAL_ESTIMADO", "TOTAL"])
    if segmento_col in df.columns:
        por_segmento = (
            df[segmento_col]
            .value_counts(dropna=False)
            .rename_axis("SEGMENTO_COMERCIAL_ESTIMADO")
            .reset_index(name="TOTAL")
        )

    funil = pd.DataFrame(columns=[eleg_col, pot_col, "TOTAL"])
    if eleg_col in df.columns and pot_col in df.columns:
        funil = (
            df.groupby([eleg_col, pot_col], dropna=False)
            .size()
            .reset_index(name="TOTAL")
            .sort_values("TOTAL", ascending=False)
        )

    qualidade_cadastro = pd.DataFrame(
        [
            {
                "INDICADOR": "CNPJ_VALIDO_PERCENTUAL",
                "VALOR": round(float(cnpj_valid.mean() * 100), 2) if not cnpj_valid.empty else 0,
            },
            {
                "INDICADOR": "TELEFONE_VALIDO_PERCENTUAL",
                "VALOR": round(float(telefone_valid.mean() * 100), 2) if not telefone_valid.empty else 0,
            },
            {
                "INDICADOR": "CEP_PREENCHIDO_PERCENTUAL",
                "VALOR": round(float((df["CEP"].astype(str).str.strip() != "").mean() * 100), 2) if "CEP" in df.columns else 0,
            },
            {
                "INDICADOR": "ENDERECO_MINIMO_PREENCHIDO_PERCENTUAL",
                "VALOR": round(
                    float(
                        (
                            (df.get("UF", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                            & (df.get("MUNICIPIO", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                            & (df.get("LOGRADOURO", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                        ).mean()
                        * 100
                    ),
                    2,
                ) if total_empresas else 0,
            },
        ]
    )

    ranking_municipio_potencial = pd.DataFrame(columns=["MUNICIPIO", "TOTAL_EMPRESAS", "VOLUME_POTENCIAL_MENSAL_ESTIMADO"])
    if municipio_col in df.columns and not volume_midpoint.empty:
        aux = df[[municipio_col]].copy()
        aux["VOLUME_POTENCIAL_MENSAL_ESTIMADO"] = volume_midpoint
        ranking_municipio_potencial = (
            aux.groupby(municipio_col, dropna=False)
            .agg(
                TOTAL_EMPRESAS=(municipio_col, "size"),
                VOLUME_POTENCIAL_MENSAL_ESTIMADO=("VOLUME_POTENCIAL_MENSAL_ESTIMADO", "sum"),
            )
            .reset_index()
            .rename(columns={municipio_col: "MUNICIPIO"})
            .sort_values(["VOLUME_POTENCIAL_MENSAL_ESTIMADO", "TOTAL_EMPRESAS"], ascending=[False, False])
            .head(20)
        )

    potencial_por_uf_segmento = pd.DataFrame(columns=["UF", "SEGMENTO_COMERCIAL_ESTIMADO", "TOTAL_EMPRESAS", "VOLUME_POTENCIAL_MENSAL_ESTIMADO"])
    if uf_col in df.columns and segmento_col in df.columns and not volume_midpoint.empty:
        aux = df[[uf_col, segmento_col]].copy()
        aux["VOLUME_POTENCIAL_MENSAL_ESTIMADO"] = volume_midpoint
        potencial_por_uf_segmento = (
            aux.groupby([uf_col, segmento_col], dropna=False)
            .agg(
                TOTAL_EMPRESAS=(uf_col, "size"),
                VOLUME_POTENCIAL_MENSAL_ESTIMADO=("VOLUME_POTENCIAL_MENSAL_ESTIMADO", "sum"),
            )
            .reset_index()
            .rename(columns={uf_col: "UF", segmento_col: "SEGMENTO_COMERCIAL_ESTIMADO"})
            .sort_values("VOLUME_POTENCIAL_MENSAL_ESTIMADO", ascending=False)
        )

    potencial_por_porte = pd.DataFrame(columns=["PORTE_LEAD_ESTIMADO", "TOTAL_EMPRESAS", "VOLUME_POTENCIAL_MENSAL_ESTIMADO"])
    if porte_col in df.columns and not volume_midpoint.empty:
        aux = df[[porte_col]].copy()
        aux["VOLUME_POTENCIAL_MENSAL_ESTIMADO"] = volume_midpoint
        potencial_por_porte = (
            aux.groupby(porte_col, dropna=False)
            .agg(
                TOTAL_EMPRESAS=(porte_col, "size"),
                VOLUME_POTENCIAL_MENSAL_ESTIMADO=("VOLUME_POTENCIAL_MENSAL_ESTIMADO", "sum"),
            )
            .reset_index()
            .rename(columns={porte_col: "PORTE_LEAD_ESTIMADO"})
            .sort_values("VOLUME_POTENCIAL_MENSAL_ESTIMADO", ascending=False)
        )

    canal_em_elegiveis = pd.DataFrame(columns=["CANAL_ABORDAGEM_SUGERIDO", "TOTAL"])
    if eleg_col in df.columns and canal_col in df.columns:
        canal_em_elegiveis = (
            df.loc[df[eleg_col] == "ELEGIVEL", canal_col]
            .value_counts(dropna=False)
            .rename_axis("CANAL_ABORDAGEM_SUGERIDO")
            .reset_index(name="TOTAL")
        )

    ingestao_diaria = pd.DataFrame(columns=["DATA_INGESTAO", "TOTAL_EMPRESAS"])
    ingestao_mensal = pd.DataFrame(columns=["MES_INGESTAO", "TOTAL_EMPRESAS"])
    funil_por_data_ingestao = pd.DataFrame(columns=["DATA_INGESTAO", "ELEGIVEIS_TOTAL", "POTENCIAL_ALTO_TOTAL", "TAXA_ELEGIVEIS"])
    qualidade_por_data_ingestao = pd.DataFrame(columns=["DATA_INGESTAO", "CNPJ_VALIDO_PERCENTUAL", "TELEFONE_VALIDO_PERCENTUAL"])
    potencial_por_data_ingestao = pd.DataFrame(columns=["DATA_INGESTAO", "VOLUME_POTENCIAL_MENSAL_ESTIMADO", "TOTAL_EMPRESAS"])

    if not data_ingestao.empty and data_ingestao.notna().any():
        aux = df.copy()
        aux["DATA_INGESTAO"] = data_ingestao.dt.date.astype(str)
        aux = aux.loc[data_ingestao.notna()].copy()

        ingestao_diaria = (
            aux.groupby("DATA_INGESTAO", dropna=False)
            .size()
            .reset_index(name="TOTAL_EMPRESAS")
            .sort_values("DATA_INGESTAO")
        )

        aux["MES_INGESTAO"] = pd.to_datetime(aux["DATA_INGESTAO"], errors="coerce").dt.to_period("M").astype(str)
        ingestao_mensal = (
            aux.groupby("MES_INGESTAO", dropna=False)
            .size()
            .reset_index(name="TOTAL_EMPRESAS")
            .sort_values("MES_INGESTAO")
        )

        if eleg_col in aux.columns and pot_col in aux.columns:
            funil_base = aux.groupby("DATA_INGESTAO", dropna=False).agg(
                TOTAL_EMPRESAS=("DATA_INGESTAO", "size"),
                ELEGIVEIS_TOTAL=(eleg_col, lambda s: (s == "ELEGIVEL").sum()),
                POTENCIAL_ALTO_TOTAL=(pot_col, lambda s: (s == "ALTO").sum()),
            ).reset_index()
            funil_base["TAXA_ELEGIVEIS"] = (
                (funil_base["ELEGIVEIS_TOTAL"] / funil_base["TOTAL_EMPRESAS"]).fillna(0) * 100
            ).round(2)
            funil_por_data_ingestao = funil_base[["DATA_INGESTAO", "ELEGIVEIS_TOTAL", "POTENCIAL_ALTO_TOTAL", "TAXA_ELEGIVEIS"]]

        cnpj_valor = is_valid_cnpj(aux[cnpj_col]) if cnpj_col in aux.columns else pd.Series([pd.NA] * len(aux))
        tel_valor = is_valid_phone(aux[telefone_col]) if telefone_col in aux.columns else pd.Series([pd.NA] * len(aux))
        quality_aux = aux[["DATA_INGESTAO"]].copy()
        quality_aux["CNPJ_VALIDO"] = cnpj_valor
        quality_aux["TELEFONE_VALIDO"] = tel_valor
        qualidade_por_data_ingestao = (
            quality_aux.groupby("DATA_INGESTAO", dropna=False)
            .agg(
                CNPJ_VALIDO_PERCENTUAL=("CNPJ_VALIDO", lambda s: round(float(s.mean() * 100), 2) if s.notna().any() else 0),
                TELEFONE_VALIDO_PERCENTUAL=("TELEFONE_VALIDO", lambda s: round(float(s.mean() * 100), 2) if s.notna().any() else 0),
            )
            .reset_index()
        )

        if not volume_midpoint.empty:
            volume_aux = aux[["DATA_INGESTAO"]].copy()
            volume_aux["VOLUME_POTENCIAL_MENSAL_ESTIMADO"] = volume_midpoint.loc[aux.index]
            potencial_por_data_ingestao = (
                volume_aux.groupby("DATA_INGESTAO", dropna=False)
                .agg(
                    VOLUME_POTENCIAL_MENSAL_ESTIMADO=("VOLUME_POTENCIAL_MENSAL_ESTIMADO", "sum"),
                    TOTAL_EMPRESAS=("DATA_INGESTAO", "size"),
                )
                .reset_index()
                .sort_values("DATA_INGESTAO")
            )

    ingestao_resumo = pd.DataFrame(
        [
            {"METRICA": "PRIMEIRA_DATA_INGESTAO", "VALOR": primeira_data_ingestao},
            {"METRICA": "ULTIMA_DATA_INGESTAO", "VALOR": ultima_data_ingestao},
        ]
    )

    return {
        "metricas_empresas_resumo": resumo,
        "metricas_empresas_por_status": status,
        "metricas_empresas_por_uf": por_uf,
        "metricas_empresas_por_segmento": por_segmento,
        "metricas_empresas_funil": funil,
        "metricas_empresas_qualidade_cadastro": qualidade_cadastro,
        "metricas_empresas_ranking_municipio_potencial": ranking_municipio_potencial,
        "metricas_empresas_potencial_por_uf_segmento": potencial_por_uf_segmento,
        "metricas_empresas_potencial_por_porte": potencial_por_porte,
        "metricas_empresas_canal_em_elegiveis": canal_em_elegiveis,
        "metricas_empresas_ingestao_resumo": ingestao_resumo,
        "metricas_empresas_ingestao_diaria": ingestao_diaria,
        "metricas_empresas_ingestao_mensal": ingestao_mensal,
        "metricas_empresas_funil_por_data_ingestao": funil_por_data_ingestao,
        "metricas_empresas_qualidade_por_data_ingestao": qualidade_por_data_ingestao,
        "metricas_empresas_potencial_por_data_ingestao": potencial_por_data_ingestao,
    }


# ==========================
# Escrita da silver/gold direto no S3
# ==========================


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer.read()

def upload_bytes_to_s3(data: bytes, bucket: str, key: str, logger: logging.Logger) -> None:
    s3 = get_s3_client()
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    logger.info("Enviado para S3: s3://%s/%s", bucket, key)


def salvar_silver_no_s3(
    df: pd.DataFrame,
    bucket: str,
    prefix: str,
    logger: logging.Logger,
) -> Dict[str, str]:
    prefix = prefix.rstrip("/")
    parquet_key = f"{prefix}/empresas_tratadas.parquet"

    parquet_bytes = dataframe_to_parquet_bytes(df)
    upload_bytes_to_s3(parquet_bytes, bucket, parquet_key, logger)

    return {"parquet": parquet_key}


def salvar_gold_no_s3(
    metricas: Dict[str, pd.DataFrame],
    bucket: str,
    prefix: str,
    logger: logging.Logger,
) -> Dict[str, Dict[str, str]]:
    prefix = prefix.rstrip("/")
    resultados: Dict[str, Dict[str, str]] = {}

    for metric_name, metric_df in metricas.items():
        dataset_prefix = f"{prefix}/{metric_name}"
        parquet_key = f"{dataset_prefix}/{metric_name}.parquet"

        parquet_bytes = dataframe_to_parquet_bytes(metric_df)
        upload_bytes_to_s3(parquet_bytes, bucket, parquet_key, logger)

        resultados[metric_name] = {
            "parquet": parquet_key,
        }

    return resultados


# ==========================
# Lambda handler
# ==========================


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:  # noqa: ANN401
    """Handler principal da AWS Lambda.

    Tudo necessario para o pipeline esta neste arquivo.

    Exemplo de evento simples (valores default batem com o setup_s3_pipeline.sh):
    {
        "bronze_bucket": "grupo4-bronze",
        "bronze_prefix": "",
        "silver_bucket": "grupo4-silver",
        "gold_bucket": "grupo4-gold",
        "silver_prefix": "empresas/",
        "gold_prefix": "metricas_empresas/"
    }
    """

    setup_logging()
    logger = get_logger("pipeline_empresas_lambda_v2")

    try:
        payload = event or {}

        # Defaults herdados do setup_s3_pipeline.sh
        bronze_bucket_default = "grupo4-bronze"
        silver_bucket_default = "grupo4-silver"
        gold_bucket_default = "grupo4-gold"

        bronze_bucket = payload.get("bronze_bucket") or os.getenv("BRONZE_BUCKET", bronze_bucket_default)
        if not bronze_bucket:
            raise ValueError("bronze_bucket deve ser informado (evento, BRONZE_BUCKET ou default grupo4-bronze)")

        bronze_key = payload.get("bronze_key") or os.getenv("BRONZE_KEY", "")
        bronze_prefix = payload.get("bronze_prefix") or os.getenv("BRONZE_PREFIX", "")

        silver_bucket = payload.get("silver_bucket") or os.getenv("SILVER_BUCKET", silver_bucket_default)
        if not silver_bucket:
            raise ValueError("silver_bucket deve ser informado (evento, SILVER_BUCKET ou default grupo4-silver)")

        gold_bucket = payload.get("gold_bucket") or os.getenv("GOLD_BUCKET", gold_bucket_default)
        if not gold_bucket:
            raise ValueError("gold_bucket deve ser informado (evento, GOLD_BUCKET ou default grupo4-gold)")

        silver_prefix = payload.get("silver_prefix") or os.getenv("SILVER_PREFIX", "empresas/")
        gold_prefix = payload.get("gold_prefix") or os.getenv("GOLD_PREFIX", "metricas_empresas/")

        logger.info(
            "Iniciando pipeline empresas na Lambda. bronze_bucket=%s, silver_bucket=%s, gold_bucket=%s",
            bronze_bucket,
            silver_bucket,
            gold_bucket,
        )

        # 1) Le bronze do S3
        empresas_bronze, _sep = read_bronze_from_s3(bronze_bucket, bronze_key, bronze_prefix, logger)

        total_empresas_bronze = len(empresas_bronze)

        # 2) Gera silver
        empresas_silver = clean_empresas_dataframe(empresas_bronze, logger)

        total_empresas_silver = len(empresas_silver)

        linhas_removidas = total_empresas_bronze - total_empresas_silver

        # 3) Salva silver no S3 (apenas parquet)
        silver_paths = salvar_silver_no_s3(
            df=empresas_silver,
            bucket=silver_bucket,
            prefix=silver_prefix,
            logger=logger,
        )

        # 4) Gera metricas gold
        gold_metrics = build_gold_metrics(empresas_silver)

        # 5) Salva metricas gold no S3 (apenas parquet)
        gold_paths = salvar_gold_no_s3(
            metricas=gold_metrics,
            bucket=gold_bucket,
            prefix=gold_prefix,
            logger=logger,
        )

        logger.info("Pipeline empresas medallion finalizado com sucesso na Lambda")

        body = {
            "message": "Pipeline empresas medallion executado com sucesso na Lambda.",
            "bronze_bucket": bronze_bucket,
            "silver_bucket": silver_bucket,
            "gold_bucket": gold_bucket,
            "silver_paths": silver_paths,
            "gold_paths": gold_paths,
            "total_empresas_bronze": total_empresas_bronze,
            "total_empresas_silver": total_empresas_silver,
            "linhas_removidas_durante_limpeza": linhas_removidas,
        }

        # Envio de notificacao de sucesso via SNS (email ou outros inscritos no topico)
        sns_topic_arn = "arn:aws:sns:us-east-1:050665165345:envio-email"

        if sns_topic_arn:
            sns_client = get_sns_client()

            mensagem_sucesso = f"""
Ola! A funcao Lambda do pipeline de empresas (medallion) foi executada com sucesso.

Detalhes da execucao:
- Bucket bronze: {bronze_bucket}
- Bucket silver: {silver_bucket}
- Bucket gold: {gold_bucket}
- Registros lidos da bronze (S3): {total_empresas_bronze}
- Registros apos tratamento (silver): {total_empresas_silver}
- Linhas afetadas/removidas na limpeza antes de gravar no S3: {linhas_removidas}

Arquivos gerados no S3:
- Silver (parquet): {json.dumps(silver_paths, ensure_ascii=False)}
- Gold (parquet): {json.dumps(gold_paths, ensure_ascii=False)}

Att,
Sistema automatizado do pipeline de empresas.
"""

            try:
                sns_client.publish(
                    TopicArn=sns_topic_arn,
                    Message=mensagem_sucesso,
                    Subject="[Notificacao Lambda] Sucesso na execucao do pipeline de empresas",
                )
                logger.info("Notificacao de sucesso enviada via SNS para o topico: %s", sns_topic_arn)
            except Exception as sns_exc:  # noqa: BLE001
                logger.exception("Falha ao enviar notificacao SNS: %s", sns_exc)
        else:
            logger.info("SNS_TOPIC_ARN nao informado no evento nem nas variaveis de ambiente; notificacao SNS nao sera enviada.")

        return {
            "statusCode": 200,
            "body": json.dumps(body, ensure_ascii=False),
        }

    except Exception as exc:  # noqa: BLE001
        logger.exception("Erro ao executar pipeline na AWS Lambda: %s", exc)
        error_body = {
            "message": "Erro ao executar pipeline na Lambda.",
            "error": str(exc),
        }
        return {
            "statusCode": 500,
            "body": json.dumps(error_body, ensure_ascii=False),
        }
