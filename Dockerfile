FROM python:3.7

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY ./ ./

ENV PYTHONPATH=/API
WORKDIR /API

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["app:app", "--host", "0.0.0.0"]