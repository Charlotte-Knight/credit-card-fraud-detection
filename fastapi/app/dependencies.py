from sqlmodel import SQLModel, Session, create_engine
from typing import Annotated
from fastapi import Depends
import sqlalchemy

hosts = ["postgres", "localhost"]
for host in hosts:
  try:
    engine = create_engine(f"postgresql://postgres:password@{host}/postgres")
    with engine.connect() as conn:
      conn.execute(sqlalchemy.text("SELECT 1"))
    break
  except Exception:
    pass


def create_db_and_tables():
  SQLModel.metadata.create_all(engine)


def get_session():
  with Session(engine) as session:
    yield session


SessionDep = Annotated[Session, Depends(get_session)]
