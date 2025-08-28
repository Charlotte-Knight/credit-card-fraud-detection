from fastapi import FastAPI

from .dependencies import create_db_and_tables, SessionDep

from .models import *

from .crud import get_crud_router

from contextlib import asynccontextmanager
from sqlmodel import delete

from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Instrumentator().instrument(app).expose(app)

app.include_router(get_crud_router(Customer, CustomerBase, CustomerPublic, "/customers"))
app.include_router(get_crud_router(Terminal, TerminalBase, TerminalPublic, "/terminals"))
app.include_router(get_crud_router(Transaction, TransactionBase, TransactionPublic, "/transactions"))
app.include_router(get_crud_router(Fraud, FraudBase, FraudPublic, "/frauds"))

@app.delete("/delete_all_data")
def delete_all_data(session: SessionDep):
  session.exec(delete(Transaction)) # type: ignore
  session.exec(delete(Fraud)) # type: ignore
  session.exec(delete(Terminal)) # type: ignore
  session.exec(delete(Customer)) # type: ignore
  session.commit()
  
@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/transactions/verify/")
def verify_transaction(transaction: TransactionBase, session: SessionDep):
  db_transaction = Transaction.model_validate(transaction)
  db_transaction.Accepted = True
  session.add(db_transaction)
  session.commit()
  session.refresh(db_transaction)
  return db_transaction