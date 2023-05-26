FROM python:3.10
WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 3030
CMD ["uvicorn", "restful_inference:app", "--host", "0.0.0.0", "--port", "3030"]