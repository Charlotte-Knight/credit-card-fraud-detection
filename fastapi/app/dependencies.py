from sqlmodel import SQLModel, Session, create_engine
from typing import Annotated
from fastapi import Depends

engine = create_engine("postgresql://postgres:password@postgres/postgres")

def create_db_and_tables():
  SQLModel.metadata.create_all(engine)

def get_session():
  with Session(engine) as session:
    yield session

SessionDep = Annotated[Session, Depends(get_session)]