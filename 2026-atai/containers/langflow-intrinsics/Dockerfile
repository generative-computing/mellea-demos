FROM langflowai/langflow:1.7.3

LABEL maintainer="Khoi-Nguyen Tran <kndtran@ibm.com>"
LABEL description="Langflow Intrinsics"

# Install additional dependencies for custom components
USER root
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
USER 1000

# Copy custom components
RUN mkdir -p /app/components
COPY components /app/components

ENV LANGFLOW_COMPONENTS_PATH=/app/components \
    LANGFLOW_AUTO_LOGIN=false \
    HOME=/app \
    LANGFLOW_CONFIG_DIR=/app/.cache/langflow

# Create writable directories for OpenShift (runs as arbitrary non-root user)
RUN mkdir -p /app/.cache/langflow /app/.mem0 /app/.composio && \
    chmod -R 777 /app/.cache /app/.mem0 /app/.composio && \
    chmod 777 /app/.venv/lib/python3.12/site-packages/langflow/alembic && \
    chmod -R 777 /app/.venv/lib/python3.12/site-packages/langflow/initial_setup/starter_projects

EXPOSE 7860

CMD ["python", "-m", "langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
