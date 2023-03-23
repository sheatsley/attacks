# Dockerfile for Adversarial Machine Learning: https://github.com/sheatsley/attacks
FROM sheatsley/models
COPY . /attacks
RUN cd /attacks && pip install --no-cache-dir -e .
