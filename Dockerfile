# Multiple Disease Prediction System -- Streamlit app image.
#
# Build (after training locally so models/ and results/ are up to date):
#   docker build -t mdps .
# Run:
#   docker run --rm -p 8501:8501 mdps
# then open http://localhost:8501

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

# libgomp1: OpenMP runtime needed by XGBoost; curl: container health check.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# Only what the app needs at runtime.
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/
COPY results/ ./results/

# Run as a non-root user.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
