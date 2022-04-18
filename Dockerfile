FROM python:3.8-slim-buster

WORKDIR /Fullsubnet_demo

COPY requirements.txt requirements.txt

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1 

RUN pip install -r requirements.txt

COPY . .

CMD [ "python", \
    "app.py", \
    "-c", \
    "config.json"]