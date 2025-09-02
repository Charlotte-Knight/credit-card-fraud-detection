from sqlmodel import SQLModel, Session, create_engine
from typing import Annotated
from fastapi import Depends
import sqlalchemy

import time


def get_engine(interval=5, retries=5):
  hosts = ["postgres", "localhost"]

  for _ in range(retries):
    for host in hosts:
      try:
        engine = create_engine(f"postgresql://postgres:password@{host}/postgres")
        with engine.connect() as conn:
          conn.execute(sqlalchemy.text("SELECT 1"))
        return engine
      except Exception:
        continue
    time.sleep(interval)

  raise ConnectionError("Could not connect to the database after several attempts")


engine = get_engine()


def create_db_and_tables():
  SQLModel.metadata.create_all(engine)


def get_session():
  with Session(engine) as session:
    yield session


SessionDep = Annotated[Session, Depends(get_session)]
