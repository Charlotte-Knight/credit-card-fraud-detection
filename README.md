# Credit Card Fraud Detection

I created this repository because I wanted to learn more about 
- Kubernetes
- Databases
- Machine Learning

and ChatGPT suggested that credit card fraud detection would be a good shout. The project so far has ended up involving Databases, APIs, and monitoring/visualisation tools like Prometheus and Grafana. 

I have implemented a [simulator](synthetic_data/app/main.py) for credit card transactions with inspiration from this practical [handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/Foreword.html). The simulator sends transactions live to an [API](fastapi/app/main.py) which puts it into a database and decides whether it has been accepted as a valid transaction, i.e. not fraud. This classification task is performed by a decision tree that is regularly trained on a recent subset of the posted transactions. 

I've configured Prometheus and Grafana to monitor the services. The dashboard looks something like this:

![Screenshot of Grafana dashboard](/docs/grafana_screenshot.png "Screenshot of Grafana dashboard")

I've tested running this with Docker Compose and also with Kubernetes (using Minikube) and the `up.sh` and `up_k8s.sh` and quick starter scripts for both approaches. In the Kubernetes setup, some simple horizontal scaling is implemented for the API.

You may have noticed in the screenshot that the performance of the classification is not fantastic, i.e. about 25% of the genuine transactions were incorrectly flagged as fraud. However, this wasn't really the point of the exercise, and it depends on the similarity of fraudulent and genuine transactions, which at the moment is configured by default to have a significant overlap. I may address this if I return to this project.