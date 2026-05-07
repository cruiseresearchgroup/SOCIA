import os
import json
import random
import logging
from typing import List, Dict, Union, Any
import matplotlib.pyplot as plt
import pandas as pd
import openai

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants for data paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/agent_society/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Load paths for all data files
amazon_data_file = os.path.join(DATA_DIR, "amazon_train_sample.json")
goodreads_data_file = os.path.join(DATA_DIR, "goodreads_train_sample.json")
yelp_data_file = os.path.join(DATA_DIR, "yelp_train_sample.json")
user_data_file = os.path.join(DATA_DIR, "user_sample.json")
item_data_file = os.path.join(DATA_DIR, "item_sample.json")
review_data_file = os.path.join(DATA_DIR, "review_sample.json")

class User:
    def __init__(self, user_id: str, preferences: Dict[str, Any], review_history: List[str], rating_tendency: float):
        self.user_id = user_id
        self.preferences = preferences
        self.review_history = review_history
        self.rating_tendency = rating_tendency

    def write_review(self, product: 'Product', llm_agent: 'ReasoningAgent') -> str:
        review = llm_agent.reason_review(self, product)
        product.receive_review(review, None)
        return review

    def rate_product(self, product: 'Product', llm_agent: 'ReasoningAgent') -> int:
        rating = llm_agent.reason_star_rating(self, product)
        product.receive_review(None, rating)
        return rating

class Product:
    def __init__(self, product_id: str, category: str, average_rating: float, review_count: int):
        self.product_id = product_id
        self.category = category
        self.average_rating = average_rating
        self.reviews = []
        self.rating_sum = 0
        self.rating_count = review_count

    def receive_review(self, review_text: Union[str, None], star_rating: Union[int, None]) -> None:
        if review_text:
            self.reviews.append(review_text)
        if star_rating is not None:
            self.rating_sum += star_rating
            self.rating_count += 1
            self.update_rating()

    def update_rating(self) -> None:
        self.average_rating = self.rating_sum / self.rating_count if self.rating_count else self.average_rating

class ReasoningAgent:
    def __init__(self, openai_key: str):
        openai.api_key = openai_key

    def reason_review(self, user: User, product: Product) -> str:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": f"User preferences: {user.preferences}\nRecent ratings: {user.review_history}\nProduct info: {product.category}, {product.average_rating}\nPlease simulate a review."
                }],
                max_tokens=150
            )
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                raise ValueError("Unexpected API response format")
        except Exception as e:
            logging.error(f"Error generating review: {e}")
            return "Could not generate review."

    def reason_star_rating(self, user: User, product: Product) -> int:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{
                    "role": "user", 
                    "content": f"User preferences: {user.preferences}\nRecent ratings: {user.review_history}\nProduct info: {product.category}, {product.average_rating}\nPlease predict a star rating."
                }],
                max_tokens=10
            )
            if response and response.choices:
                rating_str = response.choices[0].message.content.strip()
                return int(rating_str)
            else:
                raise ValueError("Unexpected API response format")
        except (ValueError, Exception) as e:
            logging.error(f"Error predicting star rating: {e}")
            return 3

class MemoryAgent:
    def __init__(self):
        self.user_data: Dict[str, Any] = {}
        self.product_data: Dict[str, Any] = {}
        self.review_data: Dict[str, Dict[str, List[str]]] = {}

    def load_user_data(self) -> None:
        if not os.path.exists(user_data_file):
            logging.error(f"User data file {user_data_file} does not exist.")
            return
        try:
            with open(user_data_file, 'r') as f:
                user_data = json.load(f)
                if isinstance(user_data, dict):
                    self.user_data = user_data.get('users', {})
                else:
                    logging.error("Error parsing user data: Expected a dictionary.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from user data file: {e}")
        except OSError as e:
            logging.error(f"OS error while loading user data: {e}")

    def load_product_data(self) -> None:
        if not os.path.exists(item_data_file):
            logging.error(f"Product data file {item_data_file} does not exist.")
            return
        try:
            with open(item_data_file, 'r') as f:
                product_data = json.load(f)
                if isinstance(product_data, dict):
                    self.product_data = product_data.get('items', {})
                else:
                    logging.error("Error parsing product data: Expected a dictionary.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from product data file: {e}")
        except OSError as e:
            logging.error(f"OS error while loading product data: {e}")

    def load_review_data(self) -> None:
        if not os.path.exists(review_data_file):
            logging.error(f"Review data file {review_data_file} does not exist.")
            return
        try:
            with open(review_data_file, 'r') as f:
                review_data = json.load(f)
                if isinstance(review_data, dict):
                    self.review_data = review_data
                else:
                    logging.error("Error parsing review data: Expected a dictionary.")
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from review data file: {e}")
        except OSError as e:
            logging.error(f"OS error while loading review data: {e}")

    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        return self.user_data.get(user_id, {})

    def get_product_profile(self, product_id: str) -> Dict[str, Any]:
        return self.product_data.get(product_id, {})

    def get_review_history(self, user_id: str, product_id: str) -> List[str]:
        return self.review_data.get(user_id, {}).get(product_id, [])

class PlanningAgent:
    def __init__(self, memory_agent: MemoryAgent):
        self.memory_agent = memory_agent

    def plan_for_review(self, user_id: str, product_id: str) -> Dict[str, Any]:
        user_profile = self.memory_agent.get_user_profile(user_id)
        product_profile = self.memory_agent.get_product_profile(product_id)
        review_history = self.memory_agent.get_review_history(user_id, product_id)
        
        return {
            "user_profile": user_profile,
            "product_profile": product_profile,
            "review_history": review_history
        }

class SimulationEnvironment:
    def __init__(self, num_users: int, num_products: int):
        self.users: List[User] = []
        self.products: List[Product] = []
        self.memory_agent = MemoryAgent()
        self.planning_agent = PlanningAgent(self.memory_agent)

        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            logging.error("OPENAI_API_KEY environment variable not set. Please set it to use the OpenAI API.")
            raise EnvironmentError("OPENAI_API_KEY environment variable not set.")

        self.llm_agent = ReasoningAgent(openai_key=openai_key)

        self.load_entities(num_users, num_products)

    def load_entities(self, num_users: int, num_products: int) -> None:
        self.memory_agent.load_user_data()
        self.memory_agent.load_product_data()
        self.memory_agent.load_review_data()

        user_data = self.memory_agent.user_data
        product_data = self.memory_agent.product_data

        for user_id, data in user_data.items():
            if len(self.users) >= num_users:
                break
            user = User(user_id, data['preferences'], data['review_history'], data.get('average_stars', 3.0))
            self.users.append(user)

        for product_id, data in product_data.items():
            if len(self.products) >= num_products:
                break
            product = Product(product_id, data['category'], data['average_rating'], data.get('review_count', 0))
            self.products.append(product)

    def run_simulation(self) -> None:
        for user in self.users:
            products_to_review = random.sample(self.products, min(len(self.products), 5))
            for product in products_to_review:
                user.write_review(product, self.llm_agent)
                user.rate_product(product, self.llm_agent)

    def visualize(self) -> None:
        ratings = [product.average_rating for product in self.products]
        plt.hist(ratings, bins=5, range=(1, 5), edgecolor='black')
        plt.title('Distribution of Average Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Products')
        plt.show()

    def save_results(self, filename: str) -> None:
        results = {
            'product_id': [product.product_id for product in self.products],
            'average_rating': [product.average_rating for product in self.products]
        }
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

def main() -> None:
    simulation = SimulationEnvironment(num_users=1000, num_products=100)
    simulation.run_simulation()
    simulation.visualize()
    simulation.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()