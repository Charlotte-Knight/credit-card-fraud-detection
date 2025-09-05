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
import time
from prometheus_client import Histogram, Counter

router = APIRouter(prefix="/fraud_detection", tags=["Fraud Detection"])

features_statement = (
  select(
    m.Transaction.Amount.label("Amount"),
    m.Transaction.Fraud.label("Fraud"),
    m.Customer.LocationX.label("CustomerLocationX"),
    m.Customer.LocationY.label("CustomerLocationY"),
    m.Terminal.LocationX.label("TerminalLocationX"),
    m.Terminal.LocationY.label("TerminalLocationY"),
    # Distance calculation (unchanged)
    func.sqrt(
      func.pow(m.Customer.LocationX - m.Terminal.LocationX, 2)
      + func.pow(m.Customer.LocationY - m.Terminal.LocationY, 2)
    ).label("Distance"),
    func.percent_rank()
    .over(partition_by=m.Transaction.CustomerID, order_by=m.Transaction.Amount)
    .label("AmountPercentile"),
  )
  .join(m.Customer, m.Customer.CustomerID == m.Transaction.CustomerID)
  .join(m.Terminal, m.Terminal.TerminalID == m.Transaction.TerminalID)
  .order_by(m.Transaction.Time.desc())
  .limit(10000)
)


feature_names = ["Amount", "Distance", "AmountPercentile"]
feature_names_plot = ["£", "Dist", "£ %"]
# feature_names = ["AmountPercentile"]
# feature_names_plot = ["£ %"]


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


def plot_placeholder():
  plt.figure(figsize=(8, 6))
  plt.text(0.5, 0.5, "Tree is not trained yet", ha="center", fontsize=12)
  plt.axis("off")
  plt.savefig("tree.svg")


def plot_tree():
  plt.figure(figsize=(8, 6))
  sklearn.tree.plot_tree(
    trees[-1],
    filled=True,
    feature_names=feature_names_plot,
    class_names=["Genuine", "Fraud"],
    label="root",
    impurity=False,
    proportion=True,
    rounded=True,
    precision=2,
    fontsize=8,
  )

  legend = "\n".join(
    f"{key}: {item}" for key, item in zip(feature_names_plot, feature_names)
  )
  plt.text(0.01, 0.99, legend, fontsize=10, verticalalignment="top")

  plt.tight_layout()
  plt.savefig("tree.svg")


# Prometheus metrics for training job
training_duration = Histogram(
  "model_training_duration_seconds", "Time spent training the model"
)
training_counter = Counter(
  "model_training_total", "Total number of model training runs"
)
feature_importance_gauge = Histogram(
  "feature_importance", "Feature importance values", ["feature_name"]
)


@router.get("/train")
def train_model(session: SessionDep) -> float | None:
  start_time = time.time()
  training_counter.inc()

  features = pd.DataFrame(get_all_transaction_features(session))
  if features.empty:
    return None

  features.fillna(0, inplace=True)
  X = features[feature_names]
  y = features["Fraud"]
  tree = sklearn.tree.DecisionTreeClassifier(max_depth=3, class_weight="balanced")
  tree.fit(X, y)
  trees.append(tree)
  del trees[:-1]
  plot_tree()
  score = tree.score(X, y)

  # Record feature importances
  for feature_name, importance in zip(feature_names, tree.feature_importances_):
    feature_importance_gauge.labels(feature_name=feature_name).observe(importance)

  # Record training duration
  training_duration.observe(time.time() - start_time)
  return score


scheduler = BackgroundScheduler()
scheduler.add_job(train_model, "interval", seconds=10, args=[next(get_session())])
scheduler.start()


@router.get("/tree/visualize", response_class=FileResponse)
def visualize_tree():
  if trees:
    return "tree.svg"
  plot_placeholder()
  return "tree.svg"


@router.get("/predict/{transaction_id}")
def tree_predict(transaction_id: int, session: SessionDep) -> bool:
  transaction_details = get_transaction_features(transaction_id, session)
  X = pd.DataFrame([transaction_details])[feature_names]
  try:
    return trees[-1].predict(X)[0]
    #return trees[-1].predict_proba(X)[0][1] > 0.9
  except Exception:
    return False


@router.post("/transactions")
def validate_transaction(transaction: m.TransactionBase, session: SessionDep):
  db_transaction = m.Transaction.model_validate(transaction)
  session.add(db_transaction)
  session.flush()

  db_transaction.Accepted = not tree_predict(db_transaction.TransactionID, session)

  session.add(db_transaction)
  session.commit()
  return db_transaction
