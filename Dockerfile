FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e .
ENTRYPOINT ["fxcast"]
CMD ["--help"]
