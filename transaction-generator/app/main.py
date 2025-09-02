import numpy as np
import pandas as pd
import requests
import argparse
from apscheduler.schedulers.background import BlockingScheduler
from dataclasses import dataclass
import logging
from typing import Callable

from rich.logging import RichHandler
from rich.traceback import install

import time

install()
logging.basicConfig(
  level=logging.INFO,
  format="%(message)s",
  handlers=[RichHandler(rich_tracebacks=True)],
)
logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class TimeInfo:
  start_date: pd.Timestamp
  n_periods: int
  period_length: int


def generate_locations(n: int, r: tuple[float, float] = (0, 100)) -> pd.DataFrame:
  locations = {
    "LocationX": np.random.uniform(r[0], r[1], n),
    "LocationY": np.random.uniform(r[0], r[1], n),
  }
  return pd.DataFrame(locations).round(4)


def generate_transaction_characters(
  n: int,
  amount_mean: tuple[float, float] = (0.1, 10),
  tx_rate: tuple[float, float] = (0.5, 2),
) -> pd.DataFrame:
  characters = {
    "AmountMean": np.random.uniform(amount_mean[0], amount_mean[1], n),
    "TxRate": np.random.uniform(tx_rate[0], tx_rate[1], n),
  }
  characters["AmountStd"] = characters["AmountMean"] / 2
  return pd.DataFrame(characters).round(4)


def generate_times(rate: float, time_info: TimeInfo) -> pd.DatetimeIndex:
  n = int(time_info.n_periods * rate * 2)
  time_deltas = np.random.exponential(1 / rate, n).cumsum() * time_info.period_length
  time_deltas = time_deltas[
    time_deltas <= time_info.n_periods * time_info.period_length
  ]
  time_deltas = pd.to_timedelta(time_deltas, unit="m")
  return time_info.start_date + time_deltas


def generate_time_windows(
  rate: float, time_info: TimeInfo, duration: tuple[float, float] = (0.1, 10)
) -> pd.DataFrame:
  start_times = generate_times(rate, time_info)
  durations = (
    np.random.uniform(duration[0], duration[1], len(start_times))
    * time_info.period_length
  )
  durations = pd.to_timedelta(durations, unit="m")
  time_windows = {"StartTime": start_times, "EndTime": start_times + durations}

  return pd.DataFrame(time_windows)


def generate_customers(n_customers: int):
  logger.info("Generating %d customers", n_customers)
  locations = generate_locations(n_customers)
  tx_characters = generate_transaction_characters(n_customers)
  customers = locations.join(tx_characters).round(4)

  return customers


def generate_terminals(n_terminals: int):
  logger.info("Generating %d terminals", n_terminals)
  locations = generate_locations(n_terminals).round(4)
  return locations


def generate_transactions(character: pd.Series, time_info: TimeInfo) -> pd.DataFrame:
  times = generate_times(character.TxRate, time_info)
  Amount = np.random.normal(character.AmountMean, character.AmountStd, len(times))
  Amount = np.clip(Amount, 0.01, None).round(2)

  transactions = {"Time": times, "Amount": Amount}

  return pd.DataFrame(transactions)


def generate_frauds(
  players: pd.DataFrame, time_info: TimeInfo, fraud_rate_fraction: float = 0.01
) -> pd.DataFrame:
  fraud_rate = len(players) * fraud_rate_fraction

  time_windows = generate_time_windows(fraud_rate, time_info)
  characters = generate_transaction_characters(len(time_windows), (5, 20), (1, 4))

  frauds = time_windows.join(characters)
  frauds[players.index.name] = np.random.choice(players.index, len(frauds))

  return frauds


def get_distances(customer: pd.Series, terminals: pd.DataFrame) -> pd.Series:
  displacements = (
    terminals[["LocationX", "LocationY"]] - customer[["LocationX", "LocationY"]]
  )
  distances = np.sqrt((displacements**2).sum(axis=1))
  return pd.Series(distances, index=terminals.index)


def generate_genuine_transactions(
  customers: pd.DataFrame, terminals: pd.DataFrame, time_info: TimeInfo
) -> pd.DataFrame:
  logger.info("Generating genuine transactions")
  transactions = []

  for CustomerID, customer in customers.iterrows():
    customer_transactions = generate_transactions(customer, time_info)

    distances = get_distances(customer, terminals)
    p = np.exp(-distances / distances.std())
    p /= p.sum()
    customer_transactions["TerminalID"] = np.random.choice(
      terminals.index, len(customer_transactions), p=p
    )

    customer_transactions["CustomerID"] = CustomerID
    transactions.append(customer_transactions)

  transactions = pd.concat(transactions, ignore_index=True)
  transactions["Fraud"] = False

  return transactions


def generate_fraudulent_customer_transactions(
  frauds: pd.DataFrame, terminals: pd.DataFrame, time_info: TimeInfo
) -> pd.DataFrame:
  if frauds.empty:
    return frauds
  logger.info("Generating fraudulent customer transactions")
  fraudulent_transactions = []

  for FraudID, fraud in frauds.iterrows():
    transactions = generate_transactions(fraud, time_info)
    transactions = transactions[
      (transactions.Time >= fraud.StartTime) & (transactions.Time <= fraud.EndTime)
    ]

    transactions["CustomerID"] = fraud["CustomerID"]
    transactions["TerminalID"] = np.random.choice(terminals.index, len(transactions))
    transactions["FraudID"] = FraudID

    fraudulent_transactions.append(transactions)

  fraudulent_transactions = pd.concat(fraudulent_transactions, ignore_index=True)
  fraudulent_transactions["Fraud"] = True
  fraudulent_transactions["FraudScenario"] = 1

  return fraudulent_transactions


def generate_fraudulent_terminal_transactions(
  frauds: pd.DataFrame, customers: pd.DataFrame, time_info: TimeInfo
) -> pd.DataFrame:
  if frauds.empty:
    return frauds
  logger.info("Generating fraudulent terminal transactions")
  fraudulent_transactions = []

  for FraudID, fraud in frauds.iterrows():
    transactions = generate_transactions(fraud, time_info)
    transactions = transactions[
      (transactions.Time >= fraud.StartTime) & (transactions.Time <= fraud.EndTime)
    ]

    transactions["TerminalID"] = fraud["TerminalID"]
    transactions["CustomerID"] = np.random.choice(customers.index, len(transactions))
    transactions["FraudID"] = FraudID

    fraudulent_transactions.append(transactions)

  fraudulent_transactions = pd.concat(fraudulent_transactions, ignore_index=True)
  fraudulent_transactions["Fraud"] = True
  fraudulent_transactions["FraudScenario"] = 2

  return fraudulent_transactions


def delete_all_data(api_url):
  response = requests.delete(api_url + "delete_all_data/")
  assert response.status_code == 200, response.text


def send_table(api_url, df, path):
  response = requests.post(
    api_url + path, data=df.to_json(orient="records"), timeout=60
  )
  assert response.status_code == 201, response.text
  return pd.DataFrame(response.json())


def send_table_single(api_url, df, path):
  response = requests.post(api_url + path, data=df.to_json(), timeout=60)
  assert response.status_code == 200, response.text


def send_table_on_time(api_url, df, path):
  df["Time"] += pd.Timestamp.now(tz="UTC") - df["Time"].min()
  df["Time"] += pd.Timedelta(len(df) / 10000, "s")

  scheduler = BlockingScheduler()
  logger.info("Scheduling transactions")
  for _, row in df.iterrows():
    scheduler.add_job(
      send_table_single, "date", run_date=row.Time, args=[api_url, row, path]
    )
  logger.info("Transactions will begin at %s UTC", df.iloc[0].Time.strftime("%H:%M:%S"))
  logger.info("Transactions will end at %s UTC", df.iloc[-1].Time.strftime("%H:%M:%S"))
  scheduler.start()


def generate_send(
  generate_f: Callable, generate_args: list, api_url: str, path: str, ID_col: str
):
  df = generate_f(*generate_args)
  if not df.empty:
    df = send_table(api_url, df, path)
    df.set_index(ID_col, inplace=True)
  return df


def get_api_url(api_urls):
  for _ in range(5):
    for url in api_urls:
      try:
        response = requests.get(url + "health/", timeout=5)
        if response.status_code == 200:
          logger.info("Connected to API at %s", url)
          return url
      except requests.exceptions.RequestException:
        logger.info("Failed to connect to API at %s", url)
        continue
    time.sleep(5)

  raise ConnectionError("Could not connect to FastAPI at any known URL")


def main(
  n_customers: int,
  n_terminals: int,
  n_periods: int,
  period_length: int,
  clear_database: bool,
  seed: int,
  dump: bool,
  api_url: str,
):
  api_url = get_api_url(api_url)

  np.random.seed(seed)
  if clear_database:
    delete_all_data(api_url)

  start_date = pd.Timestamp.now(tz="UTC")
  time_info = TimeInfo(start_date, n_periods, period_length)

  customers = generate_send(
    generate_customers, [n_customers], api_url, "customers/batch/", "CustomerID"
  )
  terminals = generate_send(
    generate_terminals, [n_terminals], api_url, "terminals/batch/", "TerminalID"
  )
  customer_frauds = generate_send(
    generate_frauds, [customers, time_info], api_url, "frauds/batch/", "FraudID"
  )
  terminal_frauds = generate_send(
    generate_frauds, [terminals, time_info], api_url, "frauds/batch/", "FraudID"
  )

  genuine_transactions = generate_genuine_transactions(customers, terminals, time_info)
  customer_fraud_transactions = generate_fraudulent_customer_transactions(
    customer_frauds, terminals, time_info
  )
  terminal_fraud_transactions = generate_fraudulent_terminal_transactions(
    terminal_frauds, customers, time_info
  )

  transactions = pd.concat(
    [
      genuine_transactions,
      customer_fraud_transactions,
      terminal_fraud_transactions,
    ],
    ignore_index=True,
  )
  transactions = transactions[
    transactions.Time <= start_date + pd.Timedelta(n_periods * period_length, "m")
  ]
  logger.info("%d transactions generated", len(transactions))

  transactions.sort_values("Time", inplace=True)

  if dump:
    transactions["Time"] += (
      pd.Timestamp.now(tz="UTC")
      - transactions["Time"].min()
      - pd.Timedelta(n_periods * period_length, "m")
    )
    response = requests.put(
      api_url + "transactions/",
      data=transactions.to_json(orient="records"),
      timeout=60,
    )
    assert response.status_code == 200, response.text
  else:
    send_table_on_time(api_url, transactions, "fraud_detection/transactions")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Synthetic Data Generation")
  parser.add_argument(
    "--n-customers",
    "-c",
    type=int,
    default=5000,
    help="Number of new customers to generate",
  )
  parser.add_argument(
    "--n-terminals",
    "-t",
    type=int,
    default=5000,
    help="Number of new terminals to generate",
  )
  parser.add_argument(
    "--n-periods",
    "-n",
    type=int,
    default=10,
    help="Number of periods to generate transactions for",
  )
  parser.add_argument(
    "--period-length",
    "-l",
    type=int,
    default=2,
    help="Length of a period in minutes",
  )
  parser.add_argument(
    "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
  )
  parser.add_argument(
    "--clear-database",
    action="store_true",
    help="Clear the database before generating new data",
  )
  parser.add_argument(
    "--dump",
    action="store_true",
    help="Dump all data into the database instead of doing it live",
  )
  parser.add_argument(
    "--api-url",
    default=("http://ccfd-api:8000/", "http://localhost:8000/"),
    nargs="+",
    type=str,
    help="API URL for the FastAPI app",
  )

  args = parser.parse_args()

  main(
    args.n_customers,
    args.n_terminals,
    args.n_periods,
    args.period_length,
    args.clear_database,
    args.seed,
    args.dump,
    args.api_url,
  )
