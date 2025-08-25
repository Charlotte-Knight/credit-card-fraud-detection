# Credit Card Fraud Detection

I created this repository because I wanted to learn more about 
- Kubernetes
- Databases
- Machine Learning

and ChatGPT suggested that credit card fraud detection would be a good shout. The project so far has ended up involving Databases, APIs, and monitoring/visualisation tools like Prometheus and Grafana. 

I have implemented a [simulator](synthetic_data/app/main.py) for credit card transactions with inspiration from this practical [handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html). The simulator sends transactions live to an [API](fastapi/app/main.py) which puts it into a database and decides whether it has been accepted as a valid transaction, i.e. not fraud. At the moment, this decision is made solely on the transaction amount (which are typically higher from fraudulent transactions). I've configured Prometheus and Grafana to monitor the API and to monitor transaction statistics extracted from the database.

I've written a docker compose file so this can all be started with
```
docker compose up -d
```
and then you can access the Grafana dashboard via [localhost:3000/d/ccfd/ccfd](http://localhost:3000/d/ccfd/ccfd).

In future, I'd like to
- Implement some simple ML to choose which transactions to block
- Tidy up Grafana dashboard
- Run this with Kubernetes and play around with auto-scaling