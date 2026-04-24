import random
from locust import HttpUser, between, task, constant

queries = [
    "what is bill pay?",
    "how can i make contactless payment?",
    "what is your phone number?",
    "Do you have savings account for children?",
    "Do you have savings account for business?"
]

class ApiUser(HttpUser):
    host = "http://0.0.0.0:31415"
    wait_time = between(2, 5)

    @task
    def attempt(self):
        response = self.client.post("/query", json={
            "query": random.choice(queries),
            "pipeline_name": "query_pipeline",
            "params": {
                "Retriever": {
                    "top_k": 10
                }
            }
        })
        return response
