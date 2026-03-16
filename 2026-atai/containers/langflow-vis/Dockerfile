FROM python:3.12-slim AS builder

LABEL maintainer="Yannis Katsis <yannis.katsis@ibm.com>"
LABEL description="LangFlow Trace Visualization"

ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Runtime stage
FROM python:3.12-slim

LABEL maintainer="Yannis Katsis <yannis.katsis@ibm.com>"
LABEL description="LangFlow Trace Visualization"

ENV PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy only necessary artifacts from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

EXPOSE 8080

CMD ["bash", "app/run_prod.sh"]
