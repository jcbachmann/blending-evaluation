#!/usr/bin/env bash
# Logs the CPU package temperature every 10 s while a command runs, and stops the command if it stays at 95 °C or above for 30 s.
# Usage: watchdog.sh <log file> <command...>
LOG="$1"; shift
"$@" & PID=$!
HOT=0
while kill -0 "$PID" 2>/dev/null; do
    T=$(sensors | awk '/Package id 0/ {gsub(/[+°C]/, "", $4); print int($4)}')
    echo "$(date +%T) $T" >> "$LOG"
    if [ "${T:-0}" -ge 95 ]; then HOT=$((HOT + 1)); else HOT=0; fi
    if [ "$HOT" -ge 3 ]; then echo "$(date +%T) stopping at $T °C" | tee -a "$LOG"; pkill -TERM -P "$PID"; kill -TERM "$PID"; fi
    sleep 10
done
wait "$PID"
