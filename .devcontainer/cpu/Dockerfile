FROM ubuntu:24.04
RUN apt -y update
RUN apt -y install curl git zip zstd

RUN apt -y install python3.12

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python3.12 -

WORKDIR /workspace

# Set up Poetry environment
COPY pyproject.toml poetry.lock README.md /workspace/
COPY drsd /workspace/drsd
RUN poetry install --with=dev
