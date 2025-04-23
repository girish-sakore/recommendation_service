FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV DB_HOST=postgres
ENV DB_PORT=5432
ENV DB_NAME=foodclub_production
ENV DB_USER=postgres
ENV DB_PASSWORD=yourpassword
ENV MODEL_PATH=/app/models/saved_model

CMD ["python", "-m", "api.server"]