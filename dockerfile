FROM python:3.6
RUN apt-get install -y python3 python3-pip
RUN pip install --upgrade pip
RUN mkdir /opt/app
WORKDIR /opt/app
COPY . /opt/app
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","80"]
