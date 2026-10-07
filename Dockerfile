FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
ENV PYTHONPATH=/app/src
CMD ["uvicorn", "--factory", "incident_engine.api:create_app", "--host", "0.0.0.0", "--port", "8000"]
