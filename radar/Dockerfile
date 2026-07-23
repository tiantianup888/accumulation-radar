FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY radar/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY accumulation_radar.py db.py pnl.py fingerprint.py ./
COPY feature_mining.py funding_features.py duckdb_store.py binance_client.py ./
COPY fetch_market_data.py fetch_funding.py build_symbol_meta.py ./
# 模型artifact + market.duckdb + fingerprints.json 通过 ./data:/data 卷挂载提供
ENV DB_PATH=/data/accumulation.db
ENV DUCKDB_PATH=/data/market.duckdb
ENV ML_MODEL_PATH=/data/model.txt
ENV ML_CFG_PATH=/data/model_config.json
ENV FP_LIB_PATH=/data/fingerprints.json
ENV PYTHONUNBUFFERED=1
CMD ["python3", "accumulation_radar.py", "loop"]