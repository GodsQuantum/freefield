# Freefield as a native MCP server

Freefield exposes Streamable HTTP MCP at /mcp on the existing web port. It uses the official Model Context Protocol TypeScript v2 SDK. There is no additional model host or inference runtime.

## Authentication and access control

The endpoint uses the same FREEFIELD_PASSCODE cookie, origin restriction, per-visitor history visibility, application allowlist, model availability and maximum-active-run guard as the web/API UI. Anonymous calls return HTTP 401. Do not put the passcode in URLs, prompts or documentation. For MCPProxy use stdio through Docker exec into the Freefield Web container. Its process inherits the existing Freefield credentials, calls the authenticated localhost HTTP API, and does not copy any secret into MCPProxy.

MCP tools are categorized by side effects: catalog, history, job, outputs and GPU are read-only; generate is a write operation; cancel_job is destructive. A MCPProxy profile limited to read tools must not silently acquire write access when this server is added.

## Tools

- list_apps: filter and paginate the live registry, including model readiness.
- get_app: inspect fields, accepted choices, presets and model dependencies.
- generate: start an app using those exact field IDs, optionally with uploaded images, video or audio. Returns a persistent history ID immediately.
- list_runs: inspect recent history.
- get_job: follow an asynchronous run and its output file names.
- get_output: view an image inline or obtain small audio/video base64; larger media are retrieved from the authenticated HTTP history API.
- cancel_job: cancel an active generation without deleting models or history.
- gpu_status: read GPU and queue information without waking the backend.

MCP binary upload is limited to approximately 2 MiB; large input assets use POST /api/apps/{id}/run as multipart form data. Outputs larger than 2 MB use the normal authenticated history file endpoint. Never send local filesystem paths or arbitrary remote URLs as a substitute for an authenticated upload.

## Orchestration and rollback

All work uses the existing Freefield runner. Tools do not bypass FREEFIELD_MAX_ACTIVE=1, the Cloud9 GPU-memory guard, or persistent history. Agents should consult get_app and gpu_status, then call generate, poll get_job and consume the resulting media once successful. Do not use direct ComfyUI or a parallel model library for new generation workflows.

To roll back, restore the previously tagged Freefield Web image in CT400; the MCPProxy upstream may be disabled independently. Gateway, inference runtime, model weights and database are not modified by this integration.

## MCPProxy registration

The MCPProxy upstream uses the local Docker socket already mounted in MCPProxy:

- name: freefield
- protocol: stdio
- command: docker
- arguments: exec, -i, freefield-web, node, --experimental-strip-types, --disable-warning=ExperimentalWarning, /app/server/mcp-stdio.ts

The stdio bridge runs inside the existing Web container and calls 127.0.0.1:3000. It does not bypass authentication, does not download models, and automatically sees the live Freefield registry on each call. Apply normal MCPProxy profile policy and inspect/approve only the eight known tools. If Freefield Web is replaced during an update, restart this upstream connection rather than restarting the GPU.
