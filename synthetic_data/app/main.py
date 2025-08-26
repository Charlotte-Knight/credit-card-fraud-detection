import numpy as np
import pandas as pd
import requests
import argparse
from tqdm import tqdm
from apscheduler.schedulers.background import BlockingScheduler
from dataclasses import dataclass

API_URL = "http://fastapi:80/"

@dataclass
class TimeInfo:
  start_date: pd.Timestamp
  n_periods: int
  period_length: int

def generate_locations(n: int, r : tuple[float, float] = (0,100)) -> pd.DataFrame:
  locations = {
    "LocationX": np.random.uniform(r[0], r[1], n),
    "LocationY": np.random.uniform(r[0], r[1], n)
  }
  return pd.DataFrame(locations).round(4)

def generate_transaction_characters(n: int,
                                    amount_mean: tuple[float, float] = (0.1, 10),
                                    tx_rate :    tuple[float, float] = (1.0, 10)) -> pd.DataFrame:
  characters = {
    "AmountMean": np.random.uniform(amount_mean[0], amount_mean[1], n),
    "TxRate":     np.random.uniform(    tx_rate[0],     tx_rate[1], n)
  }
  characters["AmountStd"] = characters["AmountMean"] / 2
  return pd.DataFrame(characters).round(4)

def generate_times(rate: float, time_info: TimeInfo) -> pd.Series:
  n = int((time_info.n_periods + 2) * rate)
  time_deltas = np.random.exponential(1/rate, n).cumsum() * time_info.period_length
  time_deltas = pd.to_timedelta(time_deltas, unit='m')
  return time_info.start_date + time_deltas

def generate_time_windows(rate: float, time_info: TimeInfo,
                          duration: tuple[float, float] = (0.1, 10)) -> pd.DataFrame:
  start_times = generate_times(rate, time_info)
  durations = np.random.uniform(duration[0], duration[1], len(start_times)) * time_info.period_length
  durations = pd.to_timedelta(durations, unit='m')
  time_windows = {
    "StartTime": start_times,
    "EndTime": start_times + durations
  }

  return pd.DataFrame(time_windows)

def generate_customers(n_customers: int):
  locations = generate_locations(n_customers)
  tx_characters = generate_transaction_characters(n_customers)
  customers = locations.join(tx_characters).round(4)

  return customers

def generate_terminals(n_terminals: int):
  locations = generate_locations(n_terminals).round(4)
  return locations

def generate_transactions(character: pd.Series, time_info: TimeInfo) -> pd.DataFrame:
  times = generate_times(character.TxRate, time_info)
  Amount = np.random.normal(character.AmountMean, character.AmountStd, len(times))
  Amount = np.clip(Amount, 0.01, None).round(2)

  transactions = {
    "Time": times,
    "Amount": Amount
  }

  return pd.DataFrame(transactions)

def generate_frauds(players: pd.DataFrame, time_info: TimeInfo,
                    fraud_rate_fraction: float = 0.01) -> pd.DataFrame:
  fraud_rate = len(players) * fraud_rate_fraction

  time_windows = generate_time_windows(fraud_rate, time_info)
  characters = generate_transaction_characters(len(time_windows), (5, 20), (5, 20))

  frauds = time_windows.join(characters)
  frauds[players.index.name] = np.random.choice(players.index, len(frauds))

  return frauds

def get_distances(customer: pd.Series, terminals: pd.DataFrame) -> pd.Series:
  displacements = terminals[["LocationX", "LocationY"]] - customer[["LocationX", "LocationY"]]
  distances = np.sqrt((displacements**2).sum(axis=1))
  return pd.Series(distances, index=terminals.index)

def generate_genuine_transactions(customers: pd.DataFrame, terminals: pd.DataFrame,
                                  time_info: TimeInfo) -> pd.DataFrame:
  transactions = []

  for CustomerID, customer in tqdm(customers.iterrows(), desc="Generating customer transactions",
                                   total=customers.shape[0]):
    customer_transactions = generate_transactions(customer, time_info)

    distances = get_distances(customer, terminals)
    p = np.exp(-distances / distances.std())
    p /= p.sum()
    customer_transactions["TerminalID"] = np.random.choice(terminals.index, len(customer_transactions), p=p)

    customer_transactions["CustomerID"] = CustomerID
    transactions.append(customer_transactions)

  transactions = pd.concat(transactions, ignore_index=True)
  transactions["Fraud"] = False

  return transactions

def generate_fraudulent_customer_transactions(frauds: pd.DataFrame, terminals: pd.DataFrame,
                                              time_info: TimeInfo) -> pd.DataFrame:
  fraudulent_transactions = []

  for FraudID, fraud in tqdm(frauds.iterrows(), desc="Generating fraudulent customer transactions",
                        total=frauds.shape[0]):
    transactions = generate_transactions(fraud, time_info)
    transactions = transactions[ (transactions.Time >= fraud.StartTime) & (transactions.Time <= fraud.EndTime) ]
    
    transactions["CustomerID"] = fraud["CustomerID"]
    transactions["TerminalID"] = np.random.choice(terminals.index, len(transactions))

    fraudulent_transactions.append(transactions)

  fraudulent_transactions = pd.concat(fraudulent_transactions, ignore_index=True)
  fraudulent_transactions["Fraud"] = True
  transactions["FraudScenario"] = 1


  return fraudulent_transactions

def generate_fraudulent_terminal_transactions(frauds: pd.DataFrame, customers: pd.DataFrame,
                                              time_info: TimeInfo) -> pd.DataFrame:
  fraudulent_transactions = []

  for FraudID, fraud in tqdm(frauds.iterrows(), desc="Generating fraudulent terminal transactions",
                        total=frauds.shape[0]):
    transactions = generate_transactions(fraud, time_info)
    transactions = transactions[ (transactions.Time >= fraud.StartTime) & (transactions.Time <= fraud.EndTime) ]
    
    transactions["TerminalID"] = fraud["TerminalID"]
    transactions["CustomerID"] = np.random.choice(customers.index, len(transactions))

    fraudulent_transactions.append(transactions)

  fraudulent_transactions = pd.concat(fraudulent_transactions, ignore_index=True)
  fraudulent_transactions["Fraud"] = True
  transactions["FraudScenario"] = 2

  return fraudulent_transactions

def delete_all_data():
  response = requests.delete(API_URL + "delete_all_data/")
  assert response.status_code == 200, response.text

def send_table(df, path, batch_size=5000):
  responses = []
  for i in tqdm(range(0, df.shape[0], batch_size), desc=f"Sending to {path}"):
    response = requests.post(API_URL + path,
                             data=df.iloc[i:i+batch_size].to_json(orient='records'), timeout=60)
    assert response.status_code == 200, response.text
    responses.append(pd.DataFrame(response.json()))
  return pd.concat(responses, ignore_index=True)

def send_table_single(df, path):
  response = requests.post(API_URL + path,
                           data=df.to_json(), timeout=60)
  assert response.status_code == 200, response.text

def send_table_on_time(df, path):
  df["Time"] += pd.Timestamp.now(tz='UTC') - df["Time"].min()
  df["Time"] += pd.Timedelta(len(df)/10000, 's')

  scheduler = BlockingScheduler()
  for _, row in tqdm(df.iterrows(), desc="Scheduling transactions", total=df.shape[0]):
    scheduler.add_job(send_table_single, 'date', run_date=row.Time, args=[row, path])
  print(f"Transactions will begin at {df.iloc[0].Time}")
  scheduler.start()

def generate_send(generate_f: callable, generate_args: list, path: str, ID_col: str):
  df = generate_f(*generate_args)
  df = send_table(df, path)
  df.set_index(ID_col, inplace=True)
  return df

def main(n_customers: int, n_terminals: int, n_periods: int, period_length: int,
         clear_database: bool, seed: int, dump: bool):
  np.random.seed(seed)
  if clear_database:
    delete_all_data()

  start_date = pd.Timestamp.now(tz='UTC')
  time_info = TimeInfo(start_date, n_periods, period_length)

  customers = generate_send(generate_customers, [n_customers], "customers/batch/", "CustomerID")
  terminals = generate_send(generate_terminals, [n_terminals], "terminals/batch/", "TerminalID")
  customer_frauds = generate_send(generate_frauds, [customers, time_info], "frauds/batch/", "FraudID")
  terminal_frauds = generate_send(generate_frauds, [terminals, time_info], "frauds/batch/", "FraudID")

  genuine_transactions = generate_genuine_transactions(customers, terminals, time_info)
  customer_fraud_transactions = generate_fraudulent_customer_transactions(customer_frauds, terminals, time_info)
  terminal_fraud_transactions = generate_fraudulent_terminal_transactions(terminal_frauds, customers, time_info)

  transactions = pd.concat([genuine_transactions, customer_fraud_transactions, terminal_fraud_transactions], ignore_index=True)
  transactions = transactions[transactions.Time <= start_date + pd.Timedelta(n_periods * period_length, 'm')]
  print(len(transactions), "transactions generated")

  transactions.sort_values("Time", inplace=True)

  if dump:
    transactions["Time"] += pd.Timestamp.now(tz='UTC') - transactions["Time"].min() - pd.Timedelta(n_periods * period_length, 'm')
    send_table(transactions, "transactions/batch/")
  else:
    send_table_on_time(transactions, "transactions/verify/")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Synthetic Data Generation")
  parser.add_argument("--n-customers", "-c", type=int, default=5000, help="Number of new customers to generate")
  parser.add_argument("--n-terminals", "-t", type=int, default=5000, help="Number of new terminals to generate")
  parser.add_argument("--n-periods", "-n", type=int, default=6, help="Number of periods to generate transactions for")
  parser.add_argument("--period-length", "-l", type=int, default=5, help="Length of a period in minutes")
  parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
  parser.add_argument("--clear-database", action='store_true', help="Clear the database before generating new data")
  parser.add_argument("--dump", action='store_true', help="Dump all data into the database instead of doing it live")

  args = parser.parse_args()
  main(args.n_customers, args.n_terminals, args.n_periods, args.period_length, args.seed,
       args.clear_database, args.dump)