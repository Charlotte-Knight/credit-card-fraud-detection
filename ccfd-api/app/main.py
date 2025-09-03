from fastapi import FastAPI

from .dependencies import create_db_and_tables, SessionDep

from . import models as m

from .crud import get_crud_router
from .fraud_detection import router as fraud_router

from contextlib import asynccontextmanager
from sqlmodel import text

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

app.include_router(
  get_crud_router(m.Customer, m.CustomerBase, m.CustomerPublic, "/customers")
)
app.include_router(
  get_crud_router(m.Terminal, m.TerminalBase, m.TerminalPublic, "/terminals")
)
app.include_router(
  get_crud_router(
    m.Transaction, m.TransactionBase, m.TransactionPublic, "/transactions"
  )
)
app.include_router(get_crud_router(m.Fraud, m.FraudBase, m.FraudPublic, "/frauds"))

app.include_router(fraud_router)


@app.get("/")
def root():
  return {"message": "Welcome to the Credit Card Fraud Detection API"}


@app.delete("/delete_all_data")
def delete_all_data(session: SessionDep):
  session.exec(text("TRUNCATE TABLE Transaction CASCADE"))
  session.exec(text("TRUNCATE TABLE Fraud CASCADE"))
  session.exec(text("TRUNCATE TABLE Terminal CASCADE"))
  session.exec(text("TRUNCATE TABLE Customer CASCADE"))
  session.commit()


@app.get("/health")
def health_check():
  return {"status": "healthy"}
