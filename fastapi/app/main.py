from datetime import datetime
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select, delete

import random

from prometheus_fastapi_instrumentator import Instrumentator

class CustomerBase(SQLModel):
  LocationX: float = Field(ge=0, le=100)
  LocationY: float = Field(ge=0, le=100)
  AmountMean: float
  AmountStd: float
  TxRate: float
  
class Customer(CustomerBase, table=True):
  CustomerID: int | None = Field(default=None, primary_key=True)

class CustomerPublic(CustomerBase):
  CustomerID: int

class TerminalBase(SQLModel):
  LocationX: float = Field(ge=0, le=100)
  LocationY: float = Field(ge=0, le=100)

class Terminal(TerminalBase, table=True):
  TerminalID: int | None = Field(default=None, primary_key=True)

class TerminalPublic(TerminalBase):
  TerminalID: int

class TransactionBase(SQLModel):
  Time: datetime
  CustomerID: int = Field(foreign_key="customer.CustomerID")
  TerminalID: int = Field(foreign_key="terminal.TerminalID")
  Amount: float = Field(ge=0)
  Fraud: bool = False
  FraudScenario: int = 0
  Accepted: bool | None = None

class Transaction(TransactionBase, table=True):
  TransactionID: int | None = Field(default=None, primary_key=True)

class TransactionPublic(TransactionBase):
  TransactionID: int

class FraudBase(SQLModel):
  CustomerID: int | None = Field(foreign_key="customer.CustomerID", default=None)
  TerminalID: int | None = Field(foreign_key="terminal.TerminalID", default=None)
  AmountMean: float
  AmountStd: float
  TxRate: float
  StartTime: datetime
  EndTime: datetime

class Fraud(FraudBase, table=True):
  FraudID: int | None = Field(default=None, primary_key=True)

class FraudPublic(FraudBase):
  FraudID: int

engine = create_engine("postgresql://postgres:password@postgres/postgres")

def create_db_and_tables():
  SQLModel.metadata.create_all(engine)


def get_session():
  with Session(engine) as session:
    yield session

SessionDep = Annotated[Session, Depends(get_session)]
app = FastAPI()

Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
def on_startup():
  create_db_and_tables()

"""----------------------------------------------------------------------------------------------"""

@app.post("/customers/", response_model=CustomerPublic)
def create_customer(customer: CustomerBase, session: SessionDep):
  db_customer = Customer.model_validate(customer)
  session.add(db_customer)
  session.commit()
  session.refresh(db_customer)
  return db_customer

@app.post("/customers/batch/", response_model=list[CustomerPublic])
def create_customers_batch(customers: list[CustomerBase], session: SessionDep):
  db_customers = [Customer.model_validate(customer) for customer in customers]
  session.add_all(db_customers)
  session.commit()
  for customer in db_customers:
    session.refresh(customer)
  return db_customers

@app.get("/customers/", response_model=list[CustomerPublic])
def read_customers(session: SessionDep):
  customers = session.exec(select(Customer)).all()
  return customers

@app.get("/customers/{customer_id}", response_model=CustomerPublic)
def read_customer(customer_id: int, session: SessionDep):
  customer = session.get(Customer, customer_id)
  if not customer:
    raise HTTPException(status_code=404, detail="Customer not found")
  return customer

"""----------------------------------------------------------------------------------------------"""

@app.post("/terminals/", response_model=TerminalPublic)
def create_terminal(terminal: TerminalBase, session: SessionDep):
  db_terminal = Terminal.model_validate(terminal)
  session.add(db_terminal)
  session.commit()
  session.refresh(db_terminal)
  return db_terminal

@app.post("/terminals/batch/", response_model=list[TerminalPublic])
def create_terminals_batch(terminals: list[TerminalBase], session: SessionDep):
  db_terminals = [Terminal.model_validate(terminal) for terminal in terminals]
  session.add_all(db_terminals)
  session.commit()
  for terminal in db_terminals:
    session.refresh(terminal)
  return db_terminals

@app.get("/terminals/", response_model=list[TerminalPublic])
def read_terminals(session: SessionDep):
  terminals = session.exec(select(Terminal)).all()
  return terminals

@app.get("/terminals/{terminal_id}", response_model=TerminalPublic)
def read_terminal(terminal_id: int, session: SessionDep):
  terminal = session.get(Terminal, terminal_id)
  if not terminal:
    raise HTTPException(status_code=404, detail="Terminal not found")
  return terminal

"""----------------------------------------------------------------------------------------------"""

@app.post("/frauds/", response_model=FraudPublic)
def create_fraud(fraud: FraudBase, session: SessionDep):
  db_fraud = Fraud.model_validate(fraud)
  session.add(db_fraud)
  session.commit()
  session.refresh(db_fraud)
  return db_fraud

@app.post("/frauds/batch/", response_model=list[FraudPublic])
def create_frauds_batch(frauds: list[FraudBase], session: SessionDep):
  db_frauds = [Fraud.model_validate(fraud) for fraud in frauds]
  session.add_all(db_frauds)
  session.commit()
  for fraud in db_frauds:
    session.refresh(fraud)
  return db_frauds

@app.get("/frauds/", response_model=list[FraudPublic])
def read_frauds(session: SessionDep):
  frauds = session.exec(select(Fraud)).all()
  return frauds

@app.get("/frauds/{fraud_id}", response_model=FraudPublic)
def read_fraud(fraud_id: int, session: SessionDep):
  fraud = session.get(Fraud, fraud_id)
  if not fraud:
    raise HTTPException(status_code=404, detail="Fraud not found")
  return fraud

"""----------------------------------------------------------------------------------------------"""

@app.post("/transactions/", response_model=TransactionPublic)
def create_transaction(transaction: TransactionBase, session: SessionDep):
  db_transaction = Transaction.model_validate(transaction)
  session.add(db_transaction)
  session.commit()
  session.refresh(db_transaction)
  return db_transaction

@app.post("/transactions/batch/", response_model=list[TransactionPublic])
def create_transactions_batch(transactions: list[TransactionBase], session: SessionDep):
  db_transactions = [Transaction.model_validate(transaction) for transaction in transactions]
  session.add_all(db_transactions)
  session.commit()
  for transaction in db_transactions:
    session.refresh(transaction)
  return db_transactions

@app.get("/transactions/", response_model=list[TransactionPublic])
def read_transactions(
  session: SessionDep,
  start_date: datetime | None = None,
  end_date: datetime | None = None,
  customer_id: int | None = None,
  fraud: bool | None = None
):
  query = select(Transaction)
  if start_date:
    query = query.where(Transaction.TX_DATETIME >= start_date)
  if end_date:
    query = query.where(Transaction.TX_DATETIME <= end_date)
  if customer_id:
    query = query.where(Transaction.CustomerID == customer_id)
  if fraud is not None:
    query = query.where(Transaction.TX_FRAUD == fraud)

  transactions = session.exec(query).all()
  return transactions

@app.get("/transactions/locations/")
def read_transaction_locations(session: SessionDep) -> list[tuple[Transaction, Terminal]]:
  statement = select(Transaction, Terminal).where(Transaction.TerminalID == Terminal.TerminalID)
  results = session.exec(statement).all()
  return results

@app.get("/transactions/{transaction_id}", response_model=TransactionPublic)
def read_transaction(transaction_id: int, session: SessionDep):
  transaction = session.get(Transaction, transaction_id)
  if not transaction:
    raise HTTPException(status_code=404, detail="Transaction not found")
  return transaction

@app.post("/transactions/verify")
def verify_transaction(transaction: TransactionBase, session: SessionDep):
  db_transaction = Transaction.model_validate(transaction)

  db_transaction.Accepted = db_transaction.Amount < 10

  session.add(db_transaction)
  session.commit()
  session.refresh(db_transaction)
  return db_transaction

"""----------------------------------------------------------------------------------------------"""

@app.delete("/delete_all_data/")
def delete_all_data(session: SessionDep):
  session.exec(delete(Transaction))
  session.exec(delete(Fraud))
  session.exec(delete(Terminal))
  session.exec(delete(Customer))
  session.commit()
  
"""----------------------------------------------------------------------------------------------"""

@app.get("/randomint")
async def get_random_int():
  return random.randint(1, 100)

@app.get("/test")
async def test_endpoint():
  return None

@app.get("/health")
def health_check():
    return {"status": "healthy"}