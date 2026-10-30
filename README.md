# Back-end Spay

API Flask para cadastro de leads e relatórios, com um servidor MCP que expõe as
operações da API como ferramentas para clientes compatíveis com o Model Context
Protocol.

## Servidor MCP

O MCP chama a API Flask já existente. Inicie a API (`python api_leads.py`) e, em
outro terminal, instale e inicie o servidor MCP:

```powershell
python -m pip install -r requirements-mcp.txt
$env:API_BASE_URL = "http://127.0.0.1:5000"
python mcp_server.py
```

`API_BASE_URL` é opcional e assume `http://127.0.0.1:5000`. `API_TIMEOUT` também
é opcional e define o tempo limite das chamadas em segundos (padrão: `30`). O
servidor usa transporte stdio; configure o cliente MCP para executar
`python <caminho-completo>/mcp_server.py`.

### Usar o MCP como endpoint HTTP (por exemplo, no n8n)

O transporte HTTP Streamable pode ser iniciado no PowerShell:

```powershell
$env:API_BASE_URL = "http://127.0.0.1:5000"
$env:MCP_TRANSPORT = "streamable-http"
$env:MCP_HOST = "127.0.0.1"
$env:MCP_PORT = "8000"
python mcp_server.py
```

No nó MCP Client Tool do n8n, selecione **Streamable HTTP** e informe:

```text
http://127.0.0.1:8000/mcp
```

Se o n8n estiver em Docker, `127.0.0.1` aponta para o próprio contêiner. Para
acesso a partir de um contêiner local, configure `MCP_HOST` como `0.0.0.0` e use
`http://host.docker.internal:8000/mcp` no n8n. O servidor libera esse hostname
na proteção de DNS rebinding do SDK. Para n8n Cloud, o servidor MCP precisa
estar hospedado em um endereço HTTPS acessível pelo n8n; o endereço `127.0.0.1`
só funciona na própria máquina. Não exponha o servidor diretamente à internet
sem proteger o acesso.

### Ferramentas disponíveis

- `autenticar_usuario`
- `criar_lead`
- `solicitar_relatorio_vendas` e `listar_relatorios_vendas`
- `solicitar_relatorio_dados` e `listar_relatorios_dados`
- `solicitar_relatorio_crm_clientes` e `listar_relatorios_crm_clientes`
- `solicitar_relatorio_crm_leads` e `listar_relatorios_crm_leads`
- `enviar_pdf` (recebe o conteúdo do arquivo codificado em Base64)
- `consultar_ipv4_ec2`

As ferramentas de relatório validam usuário e senha com `/auth/login` antes de
chamar a operação correspondente. O cadastro de lead, o upload de PDF e a
consulta de infraestrutura seguem as regras de acesso já implementadas pela
API. O `enviar_pdf` retorna o endereço S3 informado pela API.

### Exemplo de configuração para Claude Desktop

Adapte o caminho do Python e do arquivo para o ambiente local:

```json
{
  "mcpServers": {
    "spay-leads": {
      "command": "python",
      "args": ["C:\\caminho\\para\\Back-end\\mcp_server.py"],
      "env": {
        "API_BASE_URL": "http://127.0.0.1:5000"
      }
    }
  }
}
```
