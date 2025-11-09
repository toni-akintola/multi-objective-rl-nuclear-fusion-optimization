"""
Simple Flask server to serve the Python visualization.
Run this and open http://localhost:5002 in your browser.
"""
import subprocess
import sys
from pathlib import Path
from flask import Flask, send_file, Response
import threading
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    """Serve a simple page that launches the Python visualization."""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Chamber Visualization Launcher</title>
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #0a0a0f;
                color: white;
                font-family: monospace;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
            .container {
                text-align: center;
            }
            button {
                background: #0066ff;
                color: white;
                border: none;
                padding: 20px 40px;
                font-size: 18px;
                font-family: monospace;
                cursor: pointer;
                border-radius: 8px;
                margin: 10px;
            }
            button:hover {
                background: #0055dd;
            }
            .info {
                margin-top: 20px;
                color: #aaa;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Tokamak Chamber Visualization</h1>
            <p>Click to launch the Python visualization</p>
            <button onclick="window.location.href='/launch'">Launch Visualization</button>
            <div class="info">
                <p>This will open the matplotlib window</p>
                <p>Close the window to stop</p>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/launch')
def launch():
    """Launch the Python visualization script."""
    script_path = Path(__file__).parent / "visualize_chamber_live.py"
    
    # Run the script in a subprocess
    subprocess.Popen([sys.executable, str(script_path)])
    
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Launching...</title>
        <meta http-equiv="refresh" content="2;url=/">
        <style>
            body {
                margin: 0;
                padding: 0;
                background: #0a0a0f;
                color: white;
                font-family: monospace;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
            }
        </style>
    </head>
    <body>
        <h1>Launching visualization...</h1>
        <p>A matplotlib window should open shortly.</p>
    </body>
    </html>
    '''

if __name__ == '__main__':
    print("Starting visualization launcher server...")
    print("Open http://localhost:5002 in your browser")
    print("Click the button to launch the Python visualization")
    
    # Optionally open browser automatically
    threading.Timer(1.0, lambda: webbrowser.open('http://localhost:5002')).start()
    
    app.run(host='127.0.0.1', port=5002, debug=False)

