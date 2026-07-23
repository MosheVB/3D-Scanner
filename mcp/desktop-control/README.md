# desktop-control MCP server

A minimal [Model Context Protocol](https://modelcontextprotocol.io/) server exposing **screenshot**, **mouse**, and **keyboard** tools, so an AI agent (Cursor, Claude Code, etc.) can see and operate the scanner's capture UI on the host machine.

Used in this project to let an agent run full capture sessions hands-free: watch the live preview, click the scan-region seed, trigger captures, and monitor progress.

## Tools

- `screenshot` — capture the desktop (optionally a region) as PNG
- `mouse_move` / `mouse_click` / `mouse_drag`
- `key_press` / `type_text`

## Setup

```bash
pip install -r requirements.txt
python smoke_test.py   # verifies screenshot + input injection work
```

Register it in your MCP client config, pointing at your Python environment:

```json
{
  "mcpServers": {
    "desktop-control": {
      "command": "/path/to/python",
      "args": ["/path/to/repo/mcp/desktop-control/server.py"],
      "env": { "PYTHONIOENCODING": "utf-8" }
    }
  }
}
```

Security note: this server grants full input control of the host desktop to the MCP client — run it only locally and only while you need it.
