from typing import Type

from sqlmodel import SQLModel, delete, select

from fastapi import APIRouter, HTTPException, status

from .dependencies import SessionDep


def get_crud_router(
  model: Type[SQLModel],
  model_base: Type[SQLModel],
  model_public: Type[SQLModel],
  prefix: str,
) -> APIRouter:
  router = APIRouter(prefix=prefix, tags=[model.__name__])

  @router.post("/", response_model=model_public, status_code=status.HTTP_201_CREATED)
  def add(item: model_base, session: SessionDep):  # type: ignore
    db_item = model.model_validate(item)
    session.add(db_item)
    session.commit()
    session.refresh(db_item)
    return db_item

  @router.post(
    "/batch", response_model=list[model_public], status_code=status.HTTP_201_CREATED
  )
  def add_batch(items: list[model_base], session: SessionDep):  # type: ignore
    db_items = [model.model_validate(item) for item in items]
    session.add_all(db_items)
    session.commit()
    for db_item in db_items:
      session.refresh(db_item)
    return db_items

  @router.get("/{item_id}", response_model=model_public)
  def read(item_id: int, session: SessionDep):
    item = session.get(model, item_id)
    if not item:
      raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail="Item not found"
      )
    return item

  @router.get("/", response_model=list[model_public])
  def read_all(session: SessionDep, limit: int | None = None, offset: int = 0):
    items = session.exec(select(model).limit(limit).offset(offset)).all()
    return items

  @router.put("/", response_model=list[model_public])
  def replace(items: list[model_base], session: SessionDep):  # type: ignore
    session.exec(delete(model))  # type: ignore
    db_items = [model.model_validate(item) for item in items]
    session.add_all(db_items)
    session.commit()
    db_items = session.exec(select(model)).all()
    return db_items

  return router
