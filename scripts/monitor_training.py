#!/usr/bin/env python3
"""Real-time system monitor for training process."""

import subprocess
import re
import time
import json
from pathlib import Path
from collections import deque
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import psutil

# Configuration
MONITOR_INTERVAL = 2  # seconds
HISTORY_LENGTH = 300  # keep 10 minutes of data at 2s intervals

class TrainingMonitor:
    def __init__(self, pid=None):
        self.pid = pid or self._find_python_training_process()
        self.history = {
            'timestamps': deque(maxlen=HISTORY_LENGTH),
            'cpu_percent': deque(maxlen=HISTORY_LENGTH),
            'memory_mb': deque(maxlen=HISTORY_LENGTH),
            'gpu_active': deque(maxlen=HISTORY_LENGTH),
            'gpu_power_mw': deque(maxlen=HISTORY_LENGTH),
        }
        
    def _find_python_training_process(self):
        """Find the Python training process."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'python' in proc.info['name'].lower():
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if 'train' in cmdline.lower():
                        print(f"Found training process: PID {proc.info['pid']}")
                        return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None
    
    def get_process_stats(self):
        """Get CPU and memory stats for the process."""
        try:
            proc = psutil.Process(self.pid)
            return {
                'cpu_percent': proc.cpu_percent(interval=0.1),
                'memory_mb': proc.memory_info().rss / 1024 / 1024
            }
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {'cpu_percent': 0, 'memory_mb': 0}
    
    def get_gpu_stats(self):
        """Parse powermetrics output for GPU usage."""
        try:
            result = subprocess.run(
                ['sudo', 'powermetrics', '--samplers', 'gpu_power', '-i', '1000', '-n', '1'],
                capture_output=True,
                text=True,
                timeout=3
            )
            
            gpu_active = 0
            gpu_power = 0
            
            for line in result.stdout.split('\n'):
                if 'GPU HW active residency' in line:
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        gpu_active = float(match.group(1))
                elif 'GPU Power:' in line:
                    match = re.search(r'(\d+)\s*mW', line)
                    if match:
                        gpu_power = int(match.group(1))
            
            return {'gpu_active': gpu_active, 'gpu_power_mw': gpu_power}
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
            return {'gpu_active': 0, 'gpu_power_mw': 0}
    
    def update(self):
        """Collect current stats."""
        timestamp = time.time()
        proc_stats = self.get_process_stats()
        gpu_stats = self.get_gpu_stats()
        
        self.history['timestamps'].append(timestamp)
        self.history['cpu_percent'].append(proc_stats['cpu_percent'])
        self.history['memory_mb'].append(proc_stats['memory_mb'])
        self.history['gpu_active'].append(gpu_stats['gpu_active'])
        self.history['gpu_power_mw'].append(gpu_stats['gpu_power_mw'])
    
    def get_json(self):
        """Return history as JSON."""
        return json.dumps({
            'timestamps': list(self.history['timestamps']),
            'cpu_percent': list(self.history['cpu_percent']),
            'memory_mb': list(self.history['memory_mb']),
            'gpu_active': list(self.history['gpu_active']),
            'gpu_power_mw': list(self.history['gpu_power_mw']),
        })


# Global monitor instance
monitor = None

class MonitorHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(monitor.get_json().encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def get_html(self):
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1e1e1e; color: #fff; }
        .chart { margin: 20px 0; }
        h1 { color: #4CAF50; }
    </style>
</head>
<body>
    <h1>Training Process Monitor (PID: ''' + str(monitor.pid) + ''')</h1>
    <div id="cpu-chart" class="chart"></div>
    <div id="memory-chart" class="chart"></div>
    <div id="gpu-chart" class="chart"></div>
    
    <script>
        function updateCharts() {
            fetch('/data')
                .then(response => response.json())
                .then(data => {
                    const timestamps = data.timestamps.map(t => new Date(t * 1000));
                    
                    // CPU Chart
                    Plotly.newPlot('cpu-chart', [{
                        x: timestamps,
                        y: data.cpu_percent,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'CPU %',
                        line: { color: '#2196F3' }
                    }], {
                        title: 'CPU Usage (%)',
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: '#fff' },
                        xaxis: { color: '#fff' },
                        yaxis: { color: '#fff', range: [0, 100] }
                    });
                    
                    // Memory Chart
                    Plotly.newPlot('memory-chart', [{
                        x: timestamps,
                        y: data.memory_mb,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'Memory (MB)',
                        line: { color: '#4CAF50' }
                    }], {
                        title: 'Memory Usage (MB)',
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: '#fff' },
                        xaxis: { color: '#fff' },
                        yaxis: { color: '#fff' }
                    });
                    
                    // GPU Chart
                    Plotly.newPlot('gpu-chart', [{
                        x: timestamps,
                        y: data.gpu_active,
                        type: 'scatter',
                        mode: 'lines',
                        name: 'GPU Active %',
                        line: { color: '#FF5722' }
                    }], {
                        title: 'GPU Active (%)',
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: '#fff' },
                        xaxis: { color: '#fff' },
                        yaxis: { color: '#fff', range: [0, 100] }
                    });
                });
        }
        
        // Update every 2 seconds
        updateCharts();
        setInterval(updateCharts, 2000);
    </script>
</body>
</html>
        '''
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


def monitoring_loop():
    """Background thread that collects stats."""
    while True:
        monitor.update()
        time.sleep(MONITOR_INTERVAL)


if __name__ == '__main__':
    import sys
    
    pid = int(sys.argv[1]) if len(sys.argv) > 1 else None
    monitor = TrainingMonitor(pid)
    
    if monitor.pid is None:
        print("Could not find training process. Please provide PID:")
        print("  python monitor_training.py <PID>")
        sys.exit(1)
    
    # Start monitoring thread
    thread = threading.Thread(target=monitoring_loop, daemon=True)
    thread.start()
    
    # Start web server
    PORT = 8050
    print(f"Monitoring PID {monitor.pid}")
    print(f"Open http://localhost:{PORT} in your browser")
    server = HTTPServer(('localhost', PORT), MonitorHandler)
    server.serve_forever()