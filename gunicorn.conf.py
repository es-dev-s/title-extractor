import os

# List form so Gunicorn cannot treat the address as a bare string.
# 0.0.0.0 is required: 127.0.0.1 is unreachable to the platform proxy.
port = os.environ.get("PORT", "5000").strip() or "5000"
bind = [f"0.0.0.0:{port}"]
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))
timeout = int(os.environ.get("GUNICORN_TIMEOUT", "180"))
graceful_timeout = 30
forwarded_allow_ips = "*"
accesslog = "-"
errorlog = "-"
