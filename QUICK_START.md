# Quick Start Guide

## Step 1: Start WebSocket Server

In Terminal 1:
```bash
source .venv/bin/activate
python websocket_server.py
```

You should see:
```
============================================================
🚀 Starting WebSocket server...
============================================================
📍 Server: ws://localhost:8765
🌐 Open index.html in your browser
   (or visit http://localhost:8000/index.html)
============================================================
⏳ Waiting for client connection...
   (Press Ctrl+C to stop)
```

## Step 2: Serve HTML File

In Terminal 2:
```bash
source .venv/bin/activate
python -m http.server 8000
```

Or if you don't have venv activated:
```bash
python3 -m http.server 8000
```

## Step 3: Open in Browser

Open: `http://localhost:8000/index.html`

The visualization will automatically connect to the WebSocket server and start showing the plasma!

## Troubleshooting

- **Server not starting?** Make sure websockets is installed: `pip install websockets`
- **No connection?** Check browser console (F12) for WebSocket errors
- **Port already in use?** Kill the old process: `pkill -f websocket_server.py`

