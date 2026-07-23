FROM python:3.11-slim
WORKDIR /app
COPY radar/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY accumulation_radar.py db.py fingerprint.py ./
ENV DB_PATH=/data/accumulation.db
ENV PYTHONUNBUFFERED=1
CMD ["python3", "accumulation_radar.py", "loop"]