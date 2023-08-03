# Dockerfile for Adversarial Machine Learning: https://github.com/sheatsley/attacks
FROM sheatsley/models
COPY . attacks
RUN pip install --no-cache-dir attacks/ && rm -rf attacks
