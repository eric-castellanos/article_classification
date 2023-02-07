FROM python:3.7

COPY ./requirements.txt ./

RUN pip install -r requirements.txt

COPY ./ ./

ENV PYTHONPATH=/API
WORKDIR /API

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "Title.py", "--server.port=8501", "--server.address=0.0.0.0"]