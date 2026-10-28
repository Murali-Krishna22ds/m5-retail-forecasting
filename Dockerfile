FROM python:3.12-slim
WORKDIR /app
# runtime-only deps for the dashboard (training deps are not needed to serve precomputed results + live OR-Tools)
RUN pip install --no-cache-dir numpy pandas pyarrow scipy dash plotly ortools gunicorn
COPY src ./src
COPY app ./app
COPY results ./results
COPY data/processed ./data/processed
ENV PORT=8050
CMD gunicorn app.app:server --bind 0.0.0.0:${PORT} --workers 1 --timeout 120
