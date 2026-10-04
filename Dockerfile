FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY gemini_web2api/ ./gemini_web2api/
COPY config.example.json ./config.json
EXPOSE 8081

# The container binds 0.0.0.0 so published ports reach it. With api_keys: []
# the guard now generates a random API key at startup, prints it in the log,
# and requires it for every request -- no --allow-insecure needed.  Set
# "api_keys" in config.json to pin your own key; pass --allow-insecure only
# if exposure is deliberate.
CMD ["python", "-m", "gemini_web2api", "--config", "/app/config.json", "--host", "0.0.0.0"]
