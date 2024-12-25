from flask import Flask, jsonify, render_template
import psutil
import json
import os
from datetime import datetime
import sqlite3
from pathlib import Path
import threading
import logging

app = Flask(__name__)

# Database setup
DB_PATH = 'data/health_monitor.db'

def init_db():
    Path('data').mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS system_health
                 (timestamp TEXT, cpu REAL, memory REAL, disk REAL)''')
    conn.commit()
    conn.close()

def store_metrics(cpu, memory, disk):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO system_health VALUES (?, ?, ?, ?)",
              (datetime.now().isoformat(), cpu, memory, disk))
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/system-health')
def system_health():
    try:
        with open('/tmp/system_stats.json', 'r') as f:
            stats = json.load(f)
    except FileNotFoundError:
        stats = {
            'cpu': psutil.cpu_percent(interval=1),
            'memory': psutil.virtual_memory().percent,
            'disk': psutil.disk_usage('/').percent
        }

    store_metrics(stats['cpu'], stats['memory'], stats['disk'])
    return jsonify(stats)

@app.route('/api/history')
def history():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM system_health ORDER BY timestamp DESC LIMIT 100")
    data = c.fetchall()
    conn.close()

    return jsonify([{
        'timestamp': row[0],
        'cpu': row[1],
        'memory': row[2],
        'disk': row[3]
    } for row in data])

# System Metrics Collector
class SystemMetricsCollector:
    def __init__(self):
        self.metrics = {}
        self._lock = threading.Lock()

    def collect_metrics(self):
        with self._lock:
            self.metrics = {
                'cpu': psutil.cpu_percent(interval=1),
                'memory': psutil.virtual_memory().percent,
                'disk': psutil.disk_usage('/').percent,
                'network': self._get_network_stats(),
                'processes': self._get_process_info()
            }
        return self.metrics

    def _get_network_stats(self):
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

    def _get_process_info(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sorted(processes, key=lambda x: x.get('cpu_percent', 0), reverse=True)[:5]

# Alert Manager
class AlertManager:
    def __init__(self, thresholds=None):
        self.thresholds = thresholds or {
            'cpu': 80,
            'memory': 90,
            'disk': 90
        }
        self.alerts = []
        self._lock = threading.Lock()

    def check_thresholds(self, metrics):
        with self._lock:
            for metric, value in metrics.items():
                if metric in self.thresholds and value > self.thresholds[metric]:
                    self._create_alert(metric, value)

    def _create_alert(self, metric, value):
        alert = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric,
            'value': value,
            'threshold': self.thresholds[metric]
        }
        self.alerts.append(alert)
        self._send_notification(alert)

    def _send_notification(self, alert):
        # Implement notification logic (email, SMS, etc.)
        logger.warning(f"Alert: {alert['metric']} usage at {alert['value']}%")

# Metrics Database Manager
class MetricsDBManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS detailed_metrics
                        (timestamp TEXT,
                         metric_name TEXT,
                         value REAL,
                         metadata TEXT)''')
            conn.commit()

    def store_detailed_metrics(self, metrics):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            timestamp = datetime.now().isoformat()
            for metric_name, value in metrics.items():
                if isinstance(value, dict):
                    metadata = json.dumps(value)
                    value = 0  # or some aggregate value
                else:
                    metadata = '{}'

                c.execute("""INSERT INTO detailed_metrics
                           VALUES (?, ?, ?, ?)""",
                        (timestamp, metric_name, value, metadata))
            conn.commit()

# Enhanced route handlers
@app.route('/api/detailed-metrics')
def detailed_metrics():
    collector = SystemMetricsCollector()
    metrics = collector.collect_metrics()

    # Store in database
    db_manager = MetricsDBManager(DB_PATH)
    db_manager.store_detailed_metrics(metrics)

    # Check for alerts
    alert_manager = AlertManager()
    alert_manager.check_thresholds(metrics)

    return jsonify(metrics)

@app.route('/api/alerts')
def get_alerts():
    alert_manager = AlertManager()
    return jsonify(alert_manager.alerts)

@app.route('/api/processes')
def get_processes():
    collector = SystemMetricsCollector()
    return jsonify(collector.collect_metrics()['processes'])

if __name__ == '__main__':
    init_db()
    # Initialize components
    metrics_collector = SystemMetricsCollector()
    alert_manager = AlertManager()
    db_manager = MetricsDBManager(DB_PATH)

    app.run(debug=True)
