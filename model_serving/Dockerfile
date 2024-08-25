FROM python:3.9-slim

RUN apt update \
    && apt install -y make

# Install dependencies
COPY requirements/ ./requirements
COPY Makefile ./Makefile
RUN make pip-tools-prod

# Copy the model
COPY models ./models

# Copy the code
COPY src ./src
COPY bentoml ./bentoml

ENV PYTHONPATH="/"

WORKDIR /bentoml

# Run the BentoML service
CMD ["bentoml", "serve", "service:Ner"]
