from __future__ import annotations

import json
import logging
import os
import re
from io import BytesIO
from typing import Any

import boto3
import pandas as pd

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


def read_silver_from_s3(silver_bucket: str, silver_key: str, silver_prefix: str, logger: logging.Logger) -> pd.DataFrame:
    if not silver_key:
        prefix = silver_prefix.rstrip("/")
        silver_key = f"{prefix}/empresas_tratadas.parquet"

    logger.info("Lendo silver no S3: s3://%s/%s", silver_bucket, silver_key)
    raw = get_s3_client().get_object(Bucket=silver_bucket, Key=silver_key)["Body"].read()
    return pd.read_parquet(BytesIO(raw))


def read_optional_crm_silver_from_s3(
    silver_bucket: str,
    crm_silver_key: str,
    crm_silver_prefix: str,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not crm_silver_key:
        prefix = crm_silver_prefix.rstrip("/")
        crm_silver_key = f"{prefix}/base_interna_crm_vendas_tratada.parquet"

    logger.info("Tentando ler base CRM na silver: s3://%s/%s", silver_bucket, crm_silver_key)
    s3 = get_s3_client()

    try:
        raw = s3.get_object(Bucket=silver_bucket, Key=crm_silver_key)["Body"].read()
        return pd.read_parquet(BytesIO(raw))
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            logger.warning("Base CRM nao encontrada na silver (%s). Seguindo sem metricas cruzadas.", crm_silver_key)
            return pd.DataFrame()
        raise


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


def normalize_cnpj_series(series: pd.Series) -> pd.Series:
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


def build_gold_metrics(empresas_silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
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

    resumo = pd.DataFrame([
        {"METRICA": "TOTAL_EMPRESAS", "VALOR": total_empresas},
        {"METRICA": "STATUS_ATIVA_TOTAL", "VALOR": int(status_ativa)},
        {"METRICA": "STATUS_ATIVA_PERCENTUAL", "VALOR": round((status_ativa / total_empresas) * 100, 2) if total_empresas else 0},
        {"METRICA": "ELEGIVEIS_TOTAL", "VALOR": int(elegiveis)},
        {"METRICA": "ELEGIVEIS_PERCENTUAL", "VALOR": round((elegiveis / total_empresas) * 100, 2) if total_empresas else 0},
        {"METRICA": "POTENCIAL_ALTO_TOTAL", "VALOR": int(potencial_alto)},
        {"METRICA": "POTENCIAL_ALTO_PERCENTUAL", "VALOR": round((potencial_alto / total_empresas) * 100, 2) if total_empresas else 0},
        {"METRICA": "PATRIMONIO_TOTAL", "VALOR": float(patrimonio.sum(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "PATRIMONIO_MEDIO", "VALOR": float(patrimonio.mean(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "PATRIMONIO_MEDIANO", "VALOR": float(patrimonio.median(skipna=True)) if not patrimonio.empty else 0},
        {"METRICA": "ATIVAS_E_ELEGIVEIS_TOTAL", "VALOR": int(ativas_e_elegiveis)},
        {"METRICA": "TAXA_ELEGIVEIS", "VALOR": round(conversao_elegiveis, 2)},
        {"METRICA": "TAXA_POTENCIAL_ALTO_SOBRE_ELEGIVEIS", "VALOR": round(conversao_potencial_alto_sobre_elegiveis, 2)},
        {"METRICA": "VOLUME_POTENCIAL_MENSAL_ESTIMADO_TOTAL", "VALOR": float(volume_midpoint.sum(skipna=True)) if not volume_midpoint.empty else 0},
        {"METRICA": "VOLUME_POTENCIAL_MENSAL_ESTIMADO_MEDIO", "VALOR": float(volume_midpoint.mean(skipna=True)) if not volume_midpoint.empty else 0},
        {"METRICA": "CNPJ_VALIDO_PERCENTUAL", "VALOR": round(float(cnpj_valid.mean() * 100), 2) if not cnpj_valid.empty else 0},
        {"METRICA": "TELEFONE_VALIDO_PERCENTUAL", "VALOR": round(float(telefone_valid.mean() * 100), 2) if not telefone_valid.empty else 0},
        {"METRICA": "LEADS_ACIONAVEIS_TOTAL", "VALOR": int(lead_acionavel.sum()) if not lead_acionavel.empty else 0},
        {"METRICA": "LEADS_ACIONAVEIS_PERCENTUAL", "VALOR": round(conversao_acionaveis, 2)},
        {"METRICA": "REGISTROS_COM_DATA_INGESTAO_TOTAL", "VALOR": int(data_ingestao.notna().sum()) if not data_ingestao.empty else 0},
    ])

    status = pd.DataFrame(columns=["STATUS", "TOTAL"])
    if status_col in df.columns:
        status = df[status_col].value_counts(dropna=False).rename_axis("STATUS").reset_index(name="TOTAL")

    por_uf = pd.DataFrame(columns=["UF", "TOTAL"])
    if uf_col in df.columns:
        por_uf = df[uf_col].value_counts(dropna=False).rename_axis("UF").reset_index(name="TOTAL")

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

    qualidade_cadastro = pd.DataFrame([
        {"INDICADOR": "CNPJ_VALIDO_PERCENTUAL", "VALOR": round(float(cnpj_valid.mean() * 100), 2) if not cnpj_valid.empty else 0},
        {"INDICADOR": "TELEFONE_VALIDO_PERCENTUAL", "VALOR": round(float(telefone_valid.mean() * 100), 2) if not telefone_valid.empty else 0},
        {"INDICADOR": "CEP_PREENCHIDO_PERCENTUAL", "VALOR": round(float((df["CEP"].astype(str).str.strip() != "").mean() * 100), 2) if "CEP" in df.columns else 0},
        {
            "INDICADOR": "ENDERECO_MINIMO_PREENCHIDO_PERCENTUAL",
            "VALOR": round(
                float(
                    (
                        (df.get("UF", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                        & (df.get("MUNICIPIO", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                        & (df.get("LOGRADOURO", pd.Series([""] * len(df))).astype(str).str.strip() != "")
                    ).mean() * 100
                ),
                2,
            ) if total_empresas else 0,
        },
    ])

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
            funil_base["TAXA_ELEGIVEIS"] = ((funil_base["ELEGIVEIS_TOTAL"] / funil_base["TOTAL_EMPRESAS"]).fillna(0) * 100).round(2)
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

    ingestao_resumo = pd.DataFrame([
        {"METRICA": "PRIMEIRA_DATA_INGESTAO", "VALOR": primeira_data_ingestao},
        {"METRICA": "ULTIMA_DATA_INGESTAO", "VALOR": ultima_data_ingestao},
    ])

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


def build_cross_metrics_empresas_crm(empresas_silver: pd.DataFrame, crm_silver: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Gera metricas cruzadas entre empresas e CRM usando CNPJ como chave."""

    empty_summary = pd.DataFrame(
        [
            {"METRICA": "TOTAL_CNPJ_UNICOS_EMPRESAS_SILVER", "VALOR": 0},
            {"METRICA": "TOTAL_CNPJ_UNICOS_CRM_SILVER", "VALOR": 0},
            {"METRICA": "TOTAL_CNPJ_EM_COMUM", "VALOR": 0},
            {"METRICA": "TAXA_MATCH_EMPRESAS_PORCENTAGEM", "VALOR": 0},
            {"METRICA": "TAXA_MATCH_CRM_PORCENTAGEM", "VALOR": 0},
        ]
    )

    if empresas_silver.empty or crm_silver.empty:
        return {
            "metricas_cruzadas_empresas_crm_resumo": empty_summary,
            "metricas_cruzadas_empresas_crm_match_por_uf": pd.DataFrame(columns=["UF", "TOTAL_CNPJ_EM_COMUM"]),
            "metricas_cruzadas_empresas_crm_match_por_status_funil_vendas": pd.DataFrame(
                columns=["STATUS_FUNIL_VENDAS", "TOTAL_CNPJ_EM_COMUM"]
            ),
            "metricas_cruzadas_empresas_crm_score_por_potencial": pd.DataFrame(
                columns=["POTENCIAL_VENDA_MAQUININHA", "SCORE_LEAD_MEDIO", "TOTAL_CNPJ_EM_COMUM"]
            ),
        }

    emp = empresas_silver.copy()
    crm = crm_silver.copy()

    emp["CNPJ_JOIN"] = normalize_cnpj_series(emp.get("CNPJ", pd.Series([""] * len(emp), index=emp.index)))
    crm["CNPJ_JOIN"] = normalize_cnpj_series(crm.get("CNPJ", pd.Series([""] * len(crm), index=crm.index)))

    emp = emp.loc[emp["CNPJ_JOIN"].str.len() > 0].copy()
    crm = crm.loc[crm["CNPJ_JOIN"].str.len() > 0].copy()

    emp_unique = set(emp["CNPJ_JOIN"].dropna().unique())
    crm_unique = set(crm["CNPJ_JOIN"].dropna().unique())
    common = emp_unique.intersection(crm_unique)

    total_emp_unique = len(emp_unique)
    total_crm_unique = len(crm_unique)
    total_common = len(common)

    taxa_match_emp = round((total_common / total_emp_unique) * 100, 2) if total_emp_unique else 0
    taxa_match_crm = round((total_common / total_crm_unique) * 100, 2) if total_crm_unique else 0

    resumo = pd.DataFrame(
        [
            {"METRICA": "TOTAL_CNPJ_UNICOS_EMPRESAS_SILVER", "VALOR": total_emp_unique},
            {"METRICA": "TOTAL_CNPJ_UNICOS_CRM_SILVER", "VALOR": total_crm_unique},
            {"METRICA": "TOTAL_CNPJ_EM_COMUM", "VALOR": total_common},
            {"METRICA": "TAXA_MATCH_EMPRESAS_PORCENTAGEM", "VALOR": taxa_match_emp},
            {"METRICA": "TAXA_MATCH_CRM_PORCENTAGEM", "VALOR": taxa_match_crm},
        ]
    )

    emp_common = emp.loc[emp["CNPJ_JOIN"].isin(common)].copy()
    crm_common = crm.loc[crm["CNPJ_JOIN"].isin(common)].copy()

    match_por_uf = pd.DataFrame(columns=["UF", "TOTAL_CNPJ_EM_COMUM"])
    if "UF" in emp_common.columns:
        match_por_uf = (
            emp_common.groupby("UF", dropna=False)["CNPJ_JOIN"]
            .nunique()
            .reset_index(name="TOTAL_CNPJ_EM_COMUM")
            .sort_values("TOTAL_CNPJ_EM_COMUM", ascending=False)
        )

    match_por_status_funil = pd.DataFrame(columns=["STATUS_FUNIL_VENDAS", "TOTAL_CNPJ_EM_COMUM"])
    if "STATUS_FUNIL_VENDAS" in crm_common.columns:
        match_por_status_funil = (
            crm_common.groupby("STATUS_FUNIL_VENDAS", dropna=False)["CNPJ_JOIN"]
            .nunique()
            .reset_index(name="TOTAL_CNPJ_EM_COMUM")
            .sort_values("TOTAL_CNPJ_EM_COMUM", ascending=False)
        )

    score_por_potencial = pd.DataFrame(columns=["POTENCIAL_VENDA_MAQUININHA", "SCORE_LEAD_MEDIO", "TOTAL_CNPJ_EM_COMUM"])
    if "POTENCIAL_VENDA_MAQUININHA" in emp_common.columns and "SCORE_LEAD" in crm_common.columns:
        joined = emp_common[["CNPJ_JOIN", "POTENCIAL_VENDA_MAQUININHA"]].drop_duplicates().merge(
            crm_common[["CNPJ_JOIN", "SCORE_LEAD"]].drop_duplicates(),
            on="CNPJ_JOIN",
            how="inner",
        )
        joined["SCORE_LEAD_NUM"] = pd.to_numeric(joined["SCORE_LEAD"], errors="coerce")
        score_por_potencial = (
            joined.groupby("POTENCIAL_VENDA_MAQUININHA", dropna=False)
            .agg(
                SCORE_LEAD_MEDIO=("SCORE_LEAD_NUM", "mean"),
                TOTAL_CNPJ_EM_COMUM=("CNPJ_JOIN", "nunique"),
            )
            .reset_index()
        )
        score_por_potencial["SCORE_LEAD_MEDIO"] = score_por_potencial["SCORE_LEAD_MEDIO"].round(2)

    return {
        "metricas_cruzadas_empresas_crm_resumo": resumo,
        "metricas_cruzadas_empresas_crm_match_por_uf": match_por_uf,
        "metricas_cruzadas_empresas_crm_match_por_status_funil_vendas": match_por_status_funil,
        "metricas_cruzadas_empresas_crm_score_por_potencial": score_por_potencial,
    }


def dataframe_to_parquet_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_parquet(buffer, index=False)
    buffer.seek(0)
    return buffer.read()


def upload_bytes_to_s3(data: bytes, bucket: str, key: str, logger: logging.Logger) -> None:
    get_s3_client().put_object(Bucket=bucket, Key=key, Body=data)
    logger.info("Enviado para S3: s3://%s/%s", bucket, key)


def salvar_gold_no_s3(metricas: dict[str, pd.DataFrame], bucket: str, prefix: str, logger: logging.Logger) -> dict[str, dict[str, str]]:
    prefix = prefix.rstrip("/")
    resultados: dict[str, dict[str, str]] = {}

    for metric_name, metric_df in metricas.items():
        dataset_prefix = f"{prefix}/{metric_name}"
        parquet_key = f"{dataset_prefix}/{metric_name}.parquet"
        upload_bytes_to_s3(dataframe_to_parquet_bytes(metric_df), bucket, parquet_key, logger)
        resultados[metric_name] = {"parquet": parquet_key}

    return resultados


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:  # noqa: ANN401
    setup_logging()
    logger = get_logger("lambda_silver_to_gold")

    payload = event or {}
    sns_topic_arn = DEFAULT_SNS_TOPIC_ARN

    try:
        silver_bucket = payload.get("silver_bucket") or os.getenv("SILVER_BUCKET", "grupo4-silver")
        silver_prefix = payload.get("silver_prefix") or os.getenv("SILVER_PREFIX", "empresas/")
        silver_key = payload.get("silver_key") or os.getenv("SILVER_KEY", "")
        crm_silver_prefix = payload.get("crm_silver_prefix") or os.getenv("CRM_SILVER_PREFIX", "crm_vendas/")
        crm_silver_key = payload.get("crm_silver_key") or os.getenv("CRM_SILVER_KEY", "")

        gold_bucket = payload.get("gold_bucket") or os.getenv("GOLD_BUCKET", "grupo4-gold")
        gold_prefix = payload.get("gold_prefix") or os.getenv("GOLD_PREFIX", "metricas_empresas/")

        empresas_silver = read_silver_from_s3(silver_bucket, silver_key, silver_prefix, logger)
        total_empresas_silver = len(empresas_silver)

        crm_silver = read_optional_crm_silver_from_s3(
            silver_bucket=silver_bucket,
            crm_silver_key=crm_silver_key,
            crm_silver_prefix=crm_silver_prefix,
            logger=logger,
        )
        total_registros_crm_silver = len(crm_silver)

        gold_metrics = build_gold_metrics(empresas_silver)
        cross_metrics = build_cross_metrics_empresas_crm(empresas_silver, crm_silver)
        all_gold_metrics = {**gold_metrics, **cross_metrics}

        gold_paths = salvar_gold_no_s3(all_gold_metrics, gold_bucket, gold_prefix, logger)

        body = {
            "message": "Pipeline silver->gold executado com sucesso.",
            "silver_bucket": silver_bucket,
            "gold_bucket": gold_bucket,
            "gold_paths": gold_paths,
            "total_empresas_silver": total_empresas_silver,
            "total_registros_crm_silver": total_registros_crm_silver,
        }

        sns_sent, sns_info = notify_sns(
            sns_topic_arn,
            "[Notificacao Lambda] Sucesso silver->gold",
            (
                "Fluxo silver->gold executado com sucesso.\n\n"
                f"Bucket silver: {silver_bucket}\n"
                f"Bucket gold: {gold_bucket}\n"
                f"Registros silver lidos: {total_empresas_silver}\n"
                f"Registros CRM silver lidos: {total_registros_crm_silver}\n"
                f"Arquivos gold: {json.dumps(gold_paths, ensure_ascii=False)}"
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
        logger.exception("Erro no fluxo silver->gold: %s", exc)

        sns_sent, sns_info = notify_sns(
            sns_topic_arn,
            "[Notificacao Lambda] Erro silver->gold",
            f"Falha no fluxo silver->gold.\n\nErro: {exc}",
            logger,
        )

        return {
            "statusCode": 500,
            "body": json.dumps(
                {
                    "message": "Erro no fluxo silver->gold.",
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
