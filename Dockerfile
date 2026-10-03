FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY gemini_web2api/ ./gemini_web2api/
COPY config.example.json ./config.json
EXPOSE 8081

# The container has to listen on 0.0.0.0 for published ports to reach it, and
# the image ships with api_keys: [] -- so the startup guard would refuse.
# Exposure is decided by port publishing instead; bind it to loopback, e.g.
#   ports: ["127.0.0.1:8081:8081"]   (docker-compose.local.yml)
#   docker run -p 127.0.0.1:8081:8081 ...
# Set api_keys in config.json and drop this flag to require authentication.
CMD ["python", "-m", "gemini_web2api", "--config", "/app/config.json", "--host", "0.0.0.0", "--allow-insecure"]
