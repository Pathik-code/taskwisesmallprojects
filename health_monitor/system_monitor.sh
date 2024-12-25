#!/bin/bash

# Configuration
THRESHOLD_CPU=80
THRESHOLD_MEM=90
THRESHOLD_DISK=90
LOG_FILE="logs/system_health.log"
EMAIL="admin@example.com"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to get CPU usage
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | awk '{print $2 + $4}'
}

# Function to get memory usage
get_memory_usage() {
    free | grep Mem | awk '{print $3/$2 * 100.0}'
}

# Function to get disk usage
get_disk_usage() {
    df -h / | awk 'NR==2 {print $5}' | tr -d '%'
}

# Function to send alert
send_alert() {
    local metric=$1
    local value=$2
    local threshold=$3
    echo "[$(date)] ALERT: $metric usage is $value% (Threshold: $threshold%)" >> "$LOG_FILE"
    # Uncomment to enable email alerts
    # echo "High $metric usage alert: $value%" | mail -s "System Alert" $EMAIL
}

# Main monitoring loop
while true; do
    CPU_USAGE=$(get_cpu_usage)
    MEM_USAGE=$(get_memory_usage)
    DISK_USAGE=$(get_disk_usage)

    # Log current stats
    echo "[$(date)] CPU: $CPU_USAGE%, Memory: $MEM_USAGE%, Disk: $DISK_USAGE%" >> "$LOG_FILE"

    # Check thresholds and send alerts
    if (( $(echo "$CPU_USAGE > $THRESHOLD_CPU" | bc -l) )); then

        send_alert "CPU" "$CPU_USAGE" "$THRESHOLD_CPU"
    fi

    if (( $(echo "$MEM_USAGE > $THRESHOLD_MEM" | bc -l) )); then
        send_alert "Memory" "$MEM_USAGE" "$THRESHOLD_MEM"
    fi

    if (( $(echo "$DISK_USAGE > $THRESHOLD_DISK" | bc -l) )); then
        send_alert "Disk" "$DISK_USAGE" "$THRESHOLD_DISK"
    fi

    # Write current stats to temp file for API
    echo "{\"cpu\": $CPU_USAGE, \"memory\": $MEM_USAGE, \"disk\": $DISK_USAGE}" > /tmp/system_stats.json

    sleep 60
done
