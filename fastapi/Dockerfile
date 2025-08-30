FROM python:3.10


WORKDIR /code


COPY ./requirements.txt /code/requirements.txt


RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt


COPY ./app /code/app

HEALTHCHECK --start-period=5s --start-interval=10s --interval=1s --timeout=2s --retries=3 CMD curl --fail http://localhost:8000/health || exit 1

CMD ["fastapi", "run", "app/main.py"]

