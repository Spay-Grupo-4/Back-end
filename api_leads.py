import os
import random
import re
import subprocess
import sys
import unicodedata
import urllib.request
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


BRONZE_BUCKET = os.getenv("BRONZE_BUCKET", "spay-g4-bronze")
BRONZE_PREFIX = os.getenv("BRONZE_PREFIX", "relatorios_formulario/")
SNS_TOPIC_ARN = os.getenv("SNS_TOPIC_ARN", "arn:aws:sns:us-east-1:065272236984:envio-email-spay-g4")
SILVER_BUCKET = os.getenv("SILVER_BUCKET", "spay-g4-silver")
SILVER_PREFIX = os.getenv("SILVER_PREFIX", "empresas/")
SILVER_KEY = os.getenv("SILVER_KEY", f"{SILVER_PREFIX.rstrip('/')}/empresas_tratadas.parquet")
GOLD_BUCKET = os.getenv("GOLD_BUCKET", "spay-g4-gold")
RELATORIOS_PREFIX = os.getenv("RELATORIOS_PREFIX", "relatorios/")
RELATORIOS_LIST_PREFIX = os.getenv("RELATORIOS_LIST_PREFIX", "relatorios/relatorios-vendas/")
RELATORIOS_DADOS_LIST_PREFIX = os.getenv("RELATORIOS_DADOS_LIST_PREFIX", "relatorios/relatorios-dados/")
RELATORIOS_CRM_CLIENTES_LIST_PREFIX = os.getenv(
    "RELATORIOS_CRM_CLIENTES_LIST_PREFIX",
    "relatorios/relatorios-clientes/",
)
RELATORIOS_CRM_LEADS_LIST_PREFIX = os.getenv(
    "RELATORIOS_CRM_LEADS_LIST_PREFIX",
    "relatorios/relatorios-leads/",
)
RELATORIOS_KEY = os.getenv("RELATORIOS_KEY", "relatorios/relatorio_metricas_vendas.pdf")
RELATORIOS_DADOS_KEY = os.getenv("RELATORIOS_DADOS_KEY", "relatorios/relatorios-dados/")
RELATORIOS_CRM_CLIENTES_KEY = os.getenv("RELATORIOS_CRM_CLIENTES_KEY", "relatorios/relatorios-clientes/")
RELATORIOS_CRM_LEADS_KEY = os.getenv("RELATORIOS_CRM_LEADS_KEY", "relatorios/relatorios-leads/")
SCRIPT_VENDAS = os.getenv("SCRIPT_VENDAS", "Script_relatorio_vendas.py")
SCRIPT_DADOS = os.getenv("SCRIPT_DADOS", "gerar_relatorio_dados.py")
SCRIPT_CRM_CLIENTES = os.getenv("SCRIPT_CRM_CLIENTES", "gerar_relatorio_crm_clientes.py")
SCRIPT_CRM_LEADS = os.getenv("SCRIPT_CRM_LEADS", "relatorio_crm_leads.py")

AUTH_CREDENTIALS = {
    "admin": {"password": "admin123", "area": "Administrador"},
    "crmleads": {"password": "leads123", "area": "CRM Leads"},
    "vendas": {"password": "vendas123", "area": "Vendas"},
    "crmclientes": {"password": "clientes123", "area": "CRM Clientes"},
    "dados": {"password": "dados123", "area": "Dados"},
}


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


def get_ec2_instance_id() -> str:
    """Busca o instance-id da EC2 via metadata."""

    url = "http://169.254.169.254/latest/meta-data/instance-id"
    with urllib.request.urlopen(url, timeout=2) as response:  # noqa: S310
        instance_id = response.read().decode("utf-8").strip()

    if not instance_id:
        raise RuntimeError("instance-id nao encontrado")

    return instance_id


def get_ec2_public_ipv4() -> str:
    """Busca o IPv4 publico da EC2 usando boto3."""

    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes EC2 na API")

    instance_id = get_ec2_instance_id()
    ec2 = boto3.client("ec2")
    response = ec2.describe_instances(InstanceIds=[instance_id])
    reservations = response.get("Reservations", [])
    instances = reservations[0].get("Instances", []) if reservations else []
    if not instances:
        raise RuntimeError("instancia nao encontrada no EC2")

    public_ip = instances[0].get("PublicIpAddress")
    if not public_ip:
        raise RuntimeError("IPv4 publico nao disponivel para a instancia")

    return public_ip


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


def normalize_login(value: str | None) -> str:
    return (value or "").strip().lower()


def validate_login(username: str | None, password: str | None) -> dict | None:
    user_key = normalize_login(username)
    if not user_key:
        return None

    record = AUTH_CREDENTIALS.get(user_key)
    if not record:
        return None

    if record["password"] != (password or ""):
        return None

    return {"username": user_key, "area": record["area"]}


def is_vendas_user(payload: dict) -> bool:
    username = normalize_login(payload.get("username"))
    area = (payload.get("area") or "").strip().lower()
    return username == "vendas" or area == "vendas"


def is_dados_user(payload: dict) -> bool:
    username = normalize_login(payload.get("username"))
    area = (payload.get("area") or "").strip().lower()
    return username == "dados" or area == "dados"


def is_crm_clientes_user(payload: dict) -> bool:
    username = normalize_login(payload.get("username"))
    area = (payload.get("area") or "").strip().lower()
    return username == "crmclientes" or area == "crm clientes"


def is_crm_leads_user(payload: dict) -> bool:
    username = normalize_login(payload.get("username"))
    area = (payload.get("area") or "").strip().lower()
    return username == "crmleads" or area == "crm leads"


def list_reports_from_s3(prefix: str, limit: int | None = None) -> list[dict]:
    s3 = get_s3_client()
    clean_prefix = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    contents = (
        item
        for page in paginator.paginate(Bucket=GOLD_BUCKET, Prefix=clean_prefix)
        for item in page.get("Contents", [])
    )
    items = [
        item
        for item in contents
        if item.get("Key") and not item["Key"].endswith("/")
    ]
    items.sort(
        key=lambda item: item.get("LastModified").timestamp()
        if item.get("LastModified")
        else 0,
        reverse=True,
    )
    if limit is not None:
        items = items[:limit]

    results = []

    for item in items:
        key = item.get("Key", "")

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": GOLD_BUCKET, "Key": key},
            ExpiresIn=3600,
        )
        results.append(
            {
                "key": key,
                "name": key.split("/")[-1],
                "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else "",
                "size": int(item.get("Size", 0)),
                "url": url,
            }
        )

    return results


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


def upload_lead_to_bronze_parquet(df_novo: pd.DataFrame) -> dict:
    """Envia o cadastro manual para o Parquet da bronze, mantendo historico."""

    if boto3 is None:
        raise RuntimeError("boto3 nao disponivel para operacoes S3 na API")

    if not BRONZE_BUCKET:
        raise RuntimeError("BRONZE_BUCKET nao definido para envio na bronze")

    df_novo = df_novo.copy()
    if "cnpj" in df_novo.columns:
        df_novo["cnpj"] = df_novo["cnpj"].apply(only_digits)

    object_name = "dados-empresas-projeto.parquet"
    key = object_name

    s3 = get_s3_client()

    try:
        resp = s3.get_object(Bucket=BRONZE_BUCKET, Key=key)
        raw = resp["Body"].read()
        df_antigo = pd.read_parquet(BytesIO(raw))
    except s3.exceptions.NoSuchKey:  # type: ignore[attr-defined]
        df_antigo = pd.DataFrame()
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404", "NotFound"}:
            df_antigo = pd.DataFrame()
        else:
            raise

    if not df_antigo.empty:
        for col in df_antigo.columns:
            if col not in df_novo.columns:
                df_novo[col] = ""
        for col in df_novo.columns:
            if col not in df_antigo.columns:
                df_antigo[col] = ""

        for col in df_antigo.columns:
            dtype = df_antigo[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                df_novo[col] = pd.to_numeric(df_novo[col], errors="coerce")
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                df_novo[col] = pd.to_datetime(df_novo[col], errors="coerce")
            else:
                df_novo[col] = df_novo[col].where(df_novo[col].notna(), "")
                df_novo[col] = df_novo[col].astype(str)

    combinado = pd.concat([df_antigo, df_novo], ignore_index=True) if not df_antigo.empty else df_novo.copy()

    buffer = BytesIO()
    combinado.to_parquet(buffer, index=False)
    buffer.seek(0)

    s3.put_object(
        Bucket=BRONZE_BUCKET,
        Key=key,
        Body=buffer.read(),
    )

    return {
        "bucket": BRONZE_BUCKET,
        "key": key,
        "linhas_enviadas": len(df_novo),
        "linhas_anteriores": len(df_antigo),
        "linhas_finais": len(combinado),
    }


@app.post("/leads")
def criar_lead():
    """Recebe um lead do front-end e envia para a bronze em CSV."""
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
        resultado = upload_lead_to_bronze_parquet(df)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao enviar lead para bronze: %s", exc)
        return (
            jsonify({
                "success": False,
                "message": f"Erro ao enviar lead para bronze: {exc}",
            }),
            500,
        )

    notify_success_to_sns(
        subject="[API Leads] Lead manual enviado com sucesso para bronze",
        message=(
            "Cadastro manual concluido com sucesso via endpoint /leads.\n\n"
            f"Bucket bronze: {resultado['bucket']}\n"
            f"Chave bronze: {resultado['key']}\n"
            f"Linhas enviadas: {resultado['linhas_enviadas']}\n"
            f"Linhas anteriores: {resultado['linhas_anteriores']}\n"
            f"Linhas finais: {resultado['linhas_finais']}\n"
            f"CNPJ: {row['cnpj']}"
        ),
    )

    return jsonify({
        "success": True,
        "message": "Lead enviado com sucesso para bronze.",
        "details": resultado,
    })


@app.post("/auth/login")
def auth_login():
    data = request.get_json(silent=True) or {}
    username = data.get("username")
    password = data.get("password")

    if not username or not password:
        return (
            jsonify({
                "success": False,
                "message": "Usuario e senha sao obrigatorios.",
            }),
            400,
        )

    auth = validate_login(username, password)
    if not auth:
        return (
            jsonify({
                "success": False,
                "message": "Usuario ou senha invalidos.",
            }),
            401,
        )

    return jsonify({
        "success": True,
        "message": "Login realizado com sucesso.",
        "user": auth,
    })


@app.post("/relatorios/solicitar")
def solicitar_relatorio_vendas():
    payload = request.get_json(silent=True) or {}
    if not is_vendas_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de Vendas."}), 403

    try:
        script_path = os.path.join(os.path.dirname(__file__), SCRIPT_VENDAS)
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify({
                "success": False,
                "message": "Falha ao gerar relatorio.",
                "error": exc.stderr or str(exc),
            }),
            500,
        )

    return jsonify({
        "success": True,
        "message": "Relatorio gerado e enviado para o S3.",
        "s3_uri": f"s3://{GOLD_BUCKET}/relatorios/relatorios-vendas/",
        "logs": result.stdout.strip(),
    })


@app.post("/relatorios/listar")
def listar_relatorios_vendas():
    payload = request.get_json(silent=True) or {}
    if not is_vendas_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de Vendas."}), 403

    try:
        reports = list_reports_from_s3(RELATORIOS_LIST_PREFIX)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao listar relatorios do S3: %s", exc)
        return jsonify({"success": False, "message": f"Erro ao listar relatorios: {exc}"}), 500

    return jsonify({"success": True, "reports": reports})


@app.post("/relatorios/dados/solicitar")
def solicitar_relatorio_dados():
    payload = request.get_json(silent=True) or {}
    if not is_dados_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de Dados."}), 403

    try:
        script_path = os.path.join(os.path.dirname(__file__), SCRIPT_DADOS)
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify({
                "success": False,
                "message": "Falha ao gerar relatorio de dados.",
                "error": exc.stderr or str(exc),
            }),
            500,
        )

    return jsonify({
        "success": True,
        "message": "Relatorio de dados gerado e enviado para o S3.",
        "s3_uri": f"s3://{GOLD_BUCKET}/{RELATORIOS_DADOS_KEY}",
        "logs": result.stdout.strip(),
    })


@app.post("/relatorios/dados/listar")
def listar_relatorios_dados():
    payload = request.get_json(silent=True) or {}
    if not is_dados_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de Dados."}), 403

    try:
        reports = list_reports_from_s3(RELATORIOS_DADOS_LIST_PREFIX)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao listar relatorios do S3: %s", exc)
        return jsonify({"success": False, "message": f"Erro ao listar relatorios: {exc}"}), 500

    return jsonify({"success": True, "reports": reports})


@app.post("/relatorios/crm/solicitar")
def solicitar_relatorio_crm_clientes():
    payload = request.get_json(silent=True) or {}
    if not is_crm_clientes_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de CRM Clientes."}), 403

    try:
        script_path = os.path.join(os.path.dirname(__file__), SCRIPT_CRM_CLIENTES)
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify({
                "success": False,
                "message": "Falha ao gerar relatorio de CRM Clientes.",
                "error": exc.stderr or str(exc),
            }),
            500,
        )

    return jsonify({
        "success": True,
        "message": "Relatorio de CRM Clientes gerado e enviado para o S3.",
        "s3_uri": f"s3://{GOLD_BUCKET}/{RELATORIOS_CRM_CLIENTES_KEY}",
        "logs": result.stdout.strip(),
    })


@app.post("/relatorios/crm/listar")
def listar_relatorios_crm_clientes():
    payload = request.get_json(silent=True) or {}
    if not is_crm_clientes_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de CRM Clientes."}), 403

    try:
        reports = list_reports_from_s3(RELATORIOS_CRM_CLIENTES_LIST_PREFIX)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao listar relatorios do S3: %s", exc)
        return jsonify({"success": False, "message": f"Erro ao listar relatorios: {exc}"}), 500

    return jsonify({"success": True, "reports": reports})


@app.post("/relatorios/crm-leads/solicitar")
def solicitar_relatorio_crm_leads():
    payload = request.get_json(silent=True) or {}
    if not is_crm_leads_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de CRM Leads."}), 403

    try:
        script_path = os.path.join(os.path.dirname(__file__), SCRIPT_CRM_LEADS)
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        return (
            jsonify({
                "success": False,
                "message": "Falha ao gerar relatorio de CRM Leads.",
                "error": exc.stderr or str(exc),
            }),
            500,
        )

    return jsonify({
        "success": True,
        "message": "Relatorio de CRM Leads gerado e enviado para o S3.",
        "s3_uri": f"s3://{GOLD_BUCKET}/{RELATORIOS_CRM_LEADS_KEY}",
        "logs": result.stdout.strip(),
    })


@app.post("/relatorios/crm-leads/listar")
def listar_relatorios_crm_leads():
    payload = request.get_json(silent=True) or {}
    if not is_crm_leads_user(payload):
        return jsonify({"success": False, "message": "Acesso restrito ao usuario de CRM Leads."}), 403

    try:
        reports = list_reports_from_s3(RELATORIOS_CRM_LEADS_LIST_PREFIX)
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao listar relatorios do S3: %s", exc)
        return jsonify({"success": False, "message": f"Erro ao listar relatorios: {exc}"}), 500

    return jsonify({"success": True, "reports": reports})


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


@app.get("/infra/ec2-ipv4")
def get_infra_ec2_ipv4():
    """Retorna o IPv4 publico da EC2 atual."""

    try:
        public_ip = get_ec2_public_ipv4()
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Falha ao obter IPv4 da EC2: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500

    return jsonify({"success": True, "ipv4": public_ip})


if __name__ == "__main__":
    # Exemplo de execucao: python api_leads.py
    app.run(host="0.0.0.0", port=5000, debug=True)
