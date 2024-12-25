# System Health Monitor

A real-time system monitoring tool with web interface.

## Features

- Real-time CPU, Memory, and Disk usage monitoring
- Historical data tracking with charts
- Alert system for threshold breaches
- Web-based dashboard

## Installation

1. Clone the repository
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Make the monitoring script executable:
   ```bash
   chmod +x system_monitor.sh
   ```

## Usage

1. Start the monitoring script:
   ```bash
   ./system_monitor.sh
   ```

2. Start the web server:
   ```bash
   python app.py
   ```

3. Access the dashboard at http://localhost:5000

## Configuration

Edit thresholds in `system_monitor.sh`:
- CPU_THRESHOLD: Default 80%
- MEM_THRESHOLD: Default 90%
- DISK_THRESHOLD: Default 90%

## License

MIT License
