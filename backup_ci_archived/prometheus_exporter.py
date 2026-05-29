from flask import Flask, request, jsonify, Response
import requests
import time
import psutil
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = Flask(__name__)

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP Requests')  
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP Request Latency')  
CPU_USAGE = Gauge('system_cpu_usage', 'CPU Usage Percentage')  
RAM_USAGE = Gauge('system_ram_usage', 'RAM Usage Percentage')  
REQUEST_SUCCESS = Counter('http_requests_success_total', 'Total Successful HTTP Requests')
REQUEST_ERROR = Counter('http_requests_errors_total', 'Total Failed HTTP Requests')
DISK_USAGE = Gauge('system_disk_usage', 'Disk Usage Percentage')
NETWORK_BYTES_SENT = Counter('system_network_bytes_sent_total', 'Total network bytes sent')
NETWORK_BYTES_RECV = Counter('system_network_bytes_recv_total', 'Total network bytes received')
PROCESS_THREADS = Gauge('system_process_threads_count', 'Number of threads used by this Python process')
SYSTEM_UPTIME = Gauge('system_uptime_seconds', 'Total time the system has been running in seconds')


@app.route('/metrics', methods=['GET'])
def metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=None)) 
    RAM_USAGE.set(psutil.virtual_memory().percent)  
    DISK_USAGE.set(psutil.disk_usage('/').percent)
    net_io = psutil.net_io_counters()
    NETWORK_BYTES_SENT._value.set(net_io.bytes_sent)
    NETWORK_BYTES_RECV._value.set(net_io.bytes_recv)
    current_process = psutil.Process(os.getpid())
    PROCESS_THREADS.set(current_process.num_threads())
    uptime = time.time() - psutil.boot_time()
    SYSTEM_UPTIME.set(uptime)
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    REQUEST_COUNT.inc()
    api_url = "http://127.0.0.1:5005/invocations"
    data = request.get_json()
    try:
        response = requests.post(api_url, json=data)
        duration = time.time() - start_time
        REQUEST_LATENCY.observe(duration)
        REQUEST_SUCCESS.inc()
        return jsonify(response.json())
    except Exception as e:
        REQUEST_ERROR.inc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)