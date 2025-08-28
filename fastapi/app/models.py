from sqlmodel import Field, SQLModel
from datetime import datetime

class Location(SQLModel):
  LocationX: float = Field(ge=0, le=100)
  LocationY: float = Field(ge=0, le=100)

class TransactionCharacter(SQLModel):
  AmountMean: float = Field(ge=0)
  AmountStd: float = Field(ge=0)
  TxRate: float = Field(ge=0)


class CustomerBase(Location, TransactionCharacter):
  pass

class Customer(CustomerBase, table=True):
  CustomerID: int | None = Field(default=None, primary_key=True)
  
class CustomerPublic(CustomerBase):
  CustomerID: int


class TerminalBase(Location):
  pass

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
  FraudScenario: int | None = None
  Accepted: bool | None = None

class Transaction(TransactionBase, table=True):
  TransactionID: int | None = Field(default=None, primary_key=True)

class TransactionPublic(TransactionBase):
  TransactionID: int

class TransactionDetails(SQLModel):
  Amount: float
  Fraud: bool
  CustomerAmountMean: float
  CustomerAmountStd: float
  CustomerLocationX: float
  CustomerLocationY: float
  TerminalLocationX: float
  TerminalLocationY: float
  Distance: float
  ZScore: float


class FraudBase(TransactionCharacter):
  CustomerID: int | None = Field(foreign_key="customer.CustomerID", default=None)
  TerminalID: int | None = Field(foreign_key="terminal.TerminalID", default=None)
  StartTime: datetime
  EndTime: datetime
  FraudScenario: int = 0

class Fraud(FraudBase, table=True):
  FraudID: int | None = Field(default=None, primary_key=True)

class FraudPublic(FraudBase):
  FraudID: int