FROM python:3.10
WORKDIR /perf

COPY requirements.txt /perf/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /perf/
