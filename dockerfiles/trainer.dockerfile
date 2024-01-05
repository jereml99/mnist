FROM nvcr.io/nvidia/pytorch:23.10-py3

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist/ mnist/
COPY data/ data/
COPY Makefile Makefile

WORKDIR /
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -e . --no-deps --no-cache-dir


ENTRYPOINT ["python", "-u", "mnist/train_model.py"]