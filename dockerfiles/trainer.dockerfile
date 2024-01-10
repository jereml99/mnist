FROM python:3.10-slim-buster

WORKDIR /
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist/ mnist/
COPY data/ data/
COPY Makefile Makefile

RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "mnist/train_model.py"]
