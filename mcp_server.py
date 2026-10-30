"""MCP server exposing the Flask leads API as tools.

Configure ``API_BASE_URL`` to point to the running Flask API. The server uses
stdio by default; set ``MCP_TRANSPORT=streamable-http`` to serve it over HTTP.
"""

import base64
import binascii
import json
import os
from typing import Annotated, Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP
from pydantic import Field


API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000").rstrip("/")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))

mcp = FastMCP("Spay Leads API")


def _decode_response(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"response_text": text}


def _api_request(method: str, path: str, payload: dict | None = None) -> dict:
    """Call the Flask API and return its JSON response with the HTTP status."""
    headers = {"Accept": "application/json"}
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"

    request = Request(
        f"{API_BASE_URL}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urlopen(request, timeout=API_TIMEOUT) as response:  # noqa: S310
            result = _decode_response(response.read())
            if isinstance(result, dict):
                result["http_status"] = response.status
                return result
            return {"http_status": response.status, "data": result}
    except HTTPError as exc:
        result = _decode_response(exc.read())
        if isinstance(result, dict):
            result["http_status"] = exc.code
            return result
        return {"http_status": exc.code, "data": result}
    except (TimeoutError, URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return {
            "success": False,
            "http_status": 0,
            "error": f"Não foi possível acessar a API em {API_BASE_URL}: {reason}",
        }


def _authenticated_post(path: str, username: str, password: str) -> dict:
    """Validate credentials through the API before calling a protected route."""
    login = _api_request(
        "POST",
        "/auth/login",
        {"username": username, "password": password},
    )
    if login.get("http_status") != 200 or not login.get("success"):
        return login

    authenticated_username = login.get("user", {}).get("username", username)
    return _api_request("POST", path, {"username": authenticated_username})


@mcp.tool()
def autenticar_usuario(
    username: Annotated[str, Field(description="Usuário cadastrado na API.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Autentica um usuário usando o endpoint POST /auth/login."""
    return _api_request("POST", "/auth/login", {"username": username, "password": password})


@mcp.tool()
def criar_lead(
    cnpj: Annotated[str, Field(description="CNPJ da empresa.")],
    lead_type: Annotated[str, Field(description="Tipo da empresa; enviado à API como 'type'.")],
    nome_social: Annotated[str, Field(description="Razão/nome social da empresa.")],
    nome_comercial: Annotated[str, Field(description="Nome comercial da empresa.")],
    telefone: str | None = None,
    cep: str | None = None,
    uf: str | None = None,
    municipio: str | None = None,
    bairro: str | None = None,
    complemento: str | None = None,
    logradouro: str | None = None,
    patrimonio_liquido: str | None = None,
    data_inicio_situacao: str | None = None,
    data_registro: str | None = None,
    segmento_comercial: str | None = None,
) -> dict:
    """Cadastra um lead (POST /leads) e o envia para o armazenamento bronze."""
    payload = {
        "cnpj": cnpj,
        "type": lead_type,
        "nomeSocial": nome_social,
        "nomeComercial": nome_comercial,
    }
    optional_fields = {
        "telefone": telefone,
        "cep": cep,
        "uf": uf,
        "municipio": municipio,
        "bairro": bairro,
        "complemento": complemento,
        "logradouro": logradouro,
        "patrimonioLiquido": patrimonio_liquido,
        "dataInicioSituacao": data_inicio_situacao,
        "dataRegistro": data_registro,
        "segmentoComercial": segmento_comercial,
    }
    payload.update({key: value for key, value in optional_fields.items() if value is not None})
    return _api_request("POST", "/leads", payload)


@mcp.tool()
def solicitar_relatorio_vendas(
    username: Annotated[str, Field(description="Usuário autorizado da área de Vendas.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Gera o relatório de Vendas (POST /relatorios/solicitar)."""
    return _authenticated_post("/relatorios/solicitar", username, password)


@mcp.tool()
def listar_relatorios_vendas(
    username: Annotated[str, Field(description="Usuário autorizado da área de Vendas.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Lista relatórios de Vendas disponíveis no S3."""
    return _authenticated_post("/relatorios/listar", username, password)


@mcp.tool()
def solicitar_relatorio_dados(
    username: Annotated[str, Field(description="Usuário autorizado da área de Dados.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Gera o relatório da área de Dados (POST /relatorios/dados/solicitar)."""
    return _authenticated_post("/relatorios/dados/solicitar", username, password)


@mcp.tool()
def listar_relatorios_dados(
    username: Annotated[str, Field(description="Usuário autorizado da área de Dados.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Lista relatórios de Dados disponíveis no S3."""
    return _authenticated_post("/relatorios/dados/listar", username, password)


@mcp.tool()
def solicitar_relatorio_crm_clientes(
    username: Annotated[str, Field(description="Usuário autorizado da área de CRM Clientes.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Gera o relatório de CRM Clientes (POST /relatorios/crm/solicitar)."""
    return _authenticated_post("/relatorios/crm/solicitar", username, password)


@mcp.tool()
def listar_relatorios_crm_clientes(
    username: Annotated[str, Field(description="Usuário autorizado da área de CRM Clientes.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Lista relatórios de CRM Clientes disponíveis no S3."""
    return _authenticated_post("/relatorios/crm/listar", username, password)


@mcp.tool()
def solicitar_relatorio_crm_leads(
    username: Annotated[str, Field(description="Usuário autorizado da área de CRM Leads.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Gera o relatório de CRM Leads (POST /relatorios/crm-leads/solicitar)."""
    return _authenticated_post("/relatorios/crm-leads/solicitar", username, password)


@mcp.tool()
def listar_relatorios_crm_leads(
    username: Annotated[str, Field(description="Usuário autorizado da área de CRM Leads.")],
    password: Annotated[str, Field(description="Senha do usuário.")],
) -> dict:
    """Lista relatórios de CRM Leads disponíveis no S3."""
    return _authenticated_post("/relatorios/crm-leads/listar", username, password)


@mcp.tool()
def enviar_pdf(
    filename: Annotated[str, Field(description="Nome do arquivo, com extensão .pdf.")],
    content_base64: Annotated[
        str,
        Field(description="Conteúdo completo do PDF codificado em Base64."),
    ],
) -> dict:
    """Envia um PDF para o endpoint POST /upload_pdf."""
    filename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    if not filename or any(char in filename for char in '\r\n"'):
        return {"success": False, "error": "Nome de arquivo inválido."}
    if not filename.lower().endswith(".pdf"):
        return {"success": False, "error": "O nome do arquivo deve terminar em .pdf."}
    try:
        pdf_bytes = base64.b64decode(content_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        return {"success": False, "error": f"Conteúdo Base64 inválido: {exc}"}
    if not pdf_bytes:
        return {"success": False, "error": "O conteúdo do PDF está vazio."}

    boundary = "----SpayMCPBoundary7MA4YWxkTrZu0gW"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        b"Content-Type: application/pdf\r\n\r\n",
        pdf_bytes,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    request = Request(
        f"{API_BASE_URL}/upload_pdf",
        data=b"".join(parts),
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=API_TIMEOUT) as response:  # noqa: S310
            result = _decode_response(response.read())
            if isinstance(result, dict):
                result["http_status"] = response.status
                return result
            return {"http_status": response.status, "data": result}
    except HTTPError as exc:
        result = _decode_response(exc.read())
        if isinstance(result, dict):
            result["http_status"] = exc.code
            return result
        return {"http_status": exc.code, "data": result}
    except (TimeoutError, URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return {
            "success": False,
            "http_status": 0,
            "error": f"Não foi possível acessar a API em {API_BASE_URL}: {reason}",
        }


@mcp.tool()
def consultar_ipv4_ec2() -> dict:
    """Consulta o IPv4 público da instância (GET /infra/ec2-ipv4)."""
    return _api_request("GET", "/infra/ec2-ipv4")


if __name__ == "__main__":
    if MCP_TRANSPORT == "stdio":
        mcp.run()
    elif MCP_TRANSPORT in {"http", "streamable-http"}:
        mcp.settings.host = MCP_HOST
        mcp.settings.port = MCP_PORT
        mcp.settings.transport_security.allowed_hosts.append(
            f"host.docker.internal:{MCP_PORT}"
        )
        mcp.settings.transport_security.allowed_origins.append(
            f"http://host.docker.internal:{MCP_PORT}"
        )
        mcp.run(transport="streamable-http")
    else:
        raise ValueError(
            "MCP_TRANSPORT deve ser 'stdio' ou 'streamable-http'."
        )
