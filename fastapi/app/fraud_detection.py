from fastapi import APIRouter
from . import models as m
from .dependencies import SessionDep, get_session
from sqlmodel import select, func
import sklearn.tree
import pandas as pd
from fastapi.responses import FileResponse
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
from apscheduler.schedulers.background import BackgroundScheduler

router = APIRouter(prefix="/fraud_detection", tags=["Fraud Detection"])

avg_subq = (
  select(
    m.Transaction.CustomerID,
    func.avg(m.Transaction.Amount).label("CustomerAmountMean"),
    func.stddev(m.Transaction.Amount).label("CustomerAmountStd"),
  )
  .group_by(m.Transaction.CustomerID)
  .subquery()
)

features_statement = (
  select(
    m.Transaction.Amount.label("Amount"),
    m.Transaction.Fraud.label("Fraud"),
    m.Customer.LocationX.label("CustomerLocationX"),
    m.Customer.LocationY.label("CustomerLocationY"),
    m.Terminal.LocationX.label("TerminalLocationX"),
    m.Terminal.LocationY.label("TerminalLocationY"),
    avg_subq.c.CustomerAmountMean.label("CustomerAmountMean"),
    avg_subq.c.CustomerAmountStd.label("CustomerAmountStd"),
    func.sqrt(
      func.pow(m.Customer.LocationX - m.Terminal.LocationX, 2)
      + func.pow(m.Customer.LocationY - m.Terminal.LocationY, 2)
    ).label("Distance"),
    (
      func.abs(m.Transaction.Amount - avg_subq.c.CustomerAmountMean)
      / func.nullif(avg_subq.c.CustomerAmountStd, 0)
    ).label("ZScore"),
  )
  .join(m.Customer, m.Customer.CustomerID == m.Transaction.CustomerID)
  .join(m.Terminal, m.Terminal.TerminalID == m.Transaction.TerminalID)
  .join(avg_subq, avg_subq.c.CustomerID == m.Transaction.CustomerID)
)


@router.get("/features/{transaction_id}", response_model=m.TransactionFeatures)
def get_transaction_features(transaction_id: int, session: SessionDep):
  statement = features_statement.where(m.Transaction.TransactionID == transaction_id)
  results = session.exec(statement).one()
  return results


@router.get("/features", response_model=list[m.TransactionFeatures])
def get_all_transaction_features(session: SessionDep):
  statement = features_statement
  results = session.exec(statement).all()
  return results


trees = []


@router.get("/train")
def train_model(session: SessionDep) -> float | None:
  features = pd.DataFrame(get_all_transaction_features(session))
  if features.empty:
    return None

  features.fillna(0, inplace=True)
  X = features[["Amount", "Distance", "ZScore"]]
  y = features["Fraud"]
  tree = sklearn.tree.DecisionTreeClassifier(max_depth=3)
  tree.fit(X, y)
  trees.append(tree)
  return tree.score(X, y)


scheduler = BackgroundScheduler()
scheduler.add_job(train_model, "interval", seconds=10, args=[next(get_session())])
scheduler.start()


@router.get("/tree/visualize", response_class=FileResponse)
def visualize_tree():
  plt.figure(figsize=(8, 4))
  try:
    sklearn.tree.plot_tree(
      trees[-1],
      filled=True,
      feature_names=["Amount", "Distance", "ZScore"],
      class_names=["Genuine", "Fraud"],
      label="root",
      impurity=False,
      proportion=True,
      rounded=True,
      precision=2,
      fontsize=10,
    )
    plt.tight_layout()
  except Exception:
    plt.text(
      0.5,
      0.5,
      "Tree is not trained yet, press Train and refresh page",
      ha="center",
    )
    plt.axis("off")
  plt.savefig("tree.svg")
  return "tree.svg"


@router.get("/predict/{transaction_id}")
def tree_predict(transaction_id: int, session: SessionDep) -> bool:
  transaction_details = get_transaction_features(transaction_id, session)
  X = pd.DataFrame([transaction_details])[["Amount", "Distance", "ZScore"]]
  try:
    return trees[-1].predict(X)[0]
  except Exception:
    return False


@router.post("/transactions")
def validate_transaction(transaction: m.TransactionBase, session: SessionDep):
  db_transaction = m.Transaction.model_validate(transaction)
  session.add(db_transaction)
  session.commit()

  db_transaction = session.exec(
    select(m.Transaction).where(
      m.Transaction.TransactionID == db_transaction.TransactionID
    )
  ).one()

  db_transaction.Accepted = not tree_predict(db_transaction.TransactionID, session)

  session.add(db_transaction)
  session.commit()
  session.refresh(db_transaction)

  return db_transaction
