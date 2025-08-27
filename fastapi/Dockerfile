FROM python:3.10


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app

HEALTHCHECK --start-period=5s --start-interval=5s --interval=5s --timeout=2s --retries=3 CMD curl --fail http://localhost:80/health || exit 1

CMD ["fastapi", "dev", "app/main.py", "--port", "80", "--host", "0.0.0.0"]

