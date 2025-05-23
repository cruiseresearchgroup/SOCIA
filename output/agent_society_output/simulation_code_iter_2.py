import os
import json
import random
import logging
import pandas as pd
import openai
from sklearn.metrics import mean_absolute_error
import numpy as np
import openai.error

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/agent_society/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Load OpenAI API key
try:
    from keys import OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
except ImportError as e:
    logging.error("Failed to load OPENAI_API_KEY from keys.py. Please ensure the file exists and is correctly set up.")
    raise e

# Load data files
def load_json_file(filename: str) -> dict:
    """Load data from a JSON file."""
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found in {DATA_DIR}.")
        raise
    except json.JSONDecodeError:
        logging.error(f"The file {filename} contains malformed JSON.")
        raise
    except IOError:
        logging.error(f"An error occurred while reading the file {filename}.")
        raise

# Load data
try:
    user_data = load_json_file('user_sample.json')
    item_data = load_json_file('item_sample.json')
    review_data = load_json_file('review_sample.json')
    amazon_data = load_json_file('amazon_train_sample.json')
    goodreads_data = load_json_file('goodreads_train_sample.json')
    yelp_data = load_json_file('yelp_train_sample.json')
except Exception as e:
    logging.error(f"Error loading data files: {e}")
    raise

# User class
class User:
    def __init__(self, user_id: str, preferences: dict, review_history: list, rating_tendency: float):
        self.id = user_id
        self.preferences = preferences
        self.review_history = review_history
        self.rating_tendency = rating_tendency

    def write_review(self, product: 'Product') -> tuple:
        """Simulates writing a review for a product."""
        review_text = self.generate_review_text(product)
        rating = self.rate_product(product)
        self.review_history.append((product.id, rating, review_text))
        return rating, review_text

    def generate_review_text(self, product: 'Product') -> str:
        """Generate review text using OpenAI's API."""
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=f"Write a review for the product {product.id} in category {product.category}.",
                max_tokens=100
            )
            review_text = response['choices'][0]['text'].strip()
        except openai.error.OpenAIError as e:
            logging.error(f"Error generating review text: {e}")
            review_text = "Error generating review text."
        return review_text

    def rate_product(self, product: 'Product') -> int:
        """Rate the product based on user preferences."""
        base_rating = random.randint(1, 5)
        if product.category in self.preferences:
            base_rating += int(self.preferences[product.category] * 0.5)
        return max(1, min(5, base_rating))

# Product class
class Product:
    def __init__(self, item_id: str, category: str, average_rating: float, review_count: int):
        self.id = item_id
        self.category = category
        self.average_rating = average_rating
        self.review_count = review_count

    def receive_review(self, rating: int) -> None:
        """Process an incoming review by updating the review count and average rating."""
        self.review_count += 1
        self.average_rating = ((self.average_rating * (self.review_count - 1)) + rating) / self.review_count

# Planning Agent
class PlanningAgent:
    def create_plan(self, user_id: str, item_id: str) -> dict:
        """Create a plan for the user to write a review for an item."""
        return {"user_id": user_id, "item_id": item_id, "action": "write_review"}

# Memory Agent
class MemoryAgent:
    def __init__(self, user_data: dict, item_data: dict):
        self.user_data = {user['user_id']: user for user in user_data}
        self.item_data = {item['item_id']: item for item in item_data}

    def retrieve_user_info(self, user_id: str) -> dict:
        """Retrieve user information based on user ID."""
        user_info = self.user_data.get(user_id)
        if not user_info:
            logging.warning(f"User {user_id} not found.")
        return user_info

    def retrieve_item_info(self, item_id: str) -> dict:
        """Retrieve item information based on item ID."""
        item_info = self.item_data.get(item_id)
        if not item_info:
            logging.warning(f"Item {item_id} not found.")
        return item_info

# Reasoning Agent
class ReasoningAgent:
    def simulate_review(self, user: User, product: Product) -> tuple:
        """Simulate a review process by a user for a product."""
        return user.write_review(product)

# Simulation class
class Simulation:
    def __init__(self, users: list, products: list, user_data: dict, item_data: dict):
        self.users = users
        self.products = products
        self.planning_agent = PlanningAgent()
        self.memory_agent = MemoryAgent(user_data, item_data)
        self.reasoning_agent = ReasoningAgent()

    def run(self, days: int = 30) -> None:
        """Run the simulation for a given number of days."""
        for day in range(days):
            products_for_review = np.random.choice(self.products, max(1, int(0.1 * len(self.products))), replace=False)
            for user in self.users:
                if random.random() < 0.1:
                    product = random.choice(products_for_review)
                    plan = self.planning_agent.create_plan(user.id, product.id)
                    user_info = self.memory_agent.retrieve_user_info(plan['user_id'])
                    item_info = self.memory_agent.retrieve_item_info(plan['item_id'])
                    if user_info and item_info:
                        rating, review = self.reasoning_agent.simulate_review(user, product)
                        product.receive_review(rating)

    def evaluate(self) -> None:
        """Evaluate the results of the simulation."""
        total_review_count = sum([product.review_count for product in self.products])
        if total_review_count > 0:
            total_rating_sum = sum([product.average_rating * product.review_count for product in self.products])
            average_rating = total_rating_sum / total_review_count
            logging.info(f"Average rating across all products: {average_rating}")
        else:
            logging.info("No reviews available for evaluation.")

    def visualize(self) -> None:
        """Visualize the results of the simulation."""
        # Placeholder for visualization logic.
        logging.info("Visualization logic is not yet implemented.")

    def save_results(self, filename: str) -> None:
        """Save the results of the simulation to a CSV file."""
        results = {"users": [user.id for user in self.users],
                   "products": [product.id for product in self.products]}
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

def main() -> None:
    if not user_data or not item_data:
        logging.error("User data or item data not loaded. Exiting.")
        return

    users = [User(user['user_id'], user.get('preferences', {}), [], user.get('average_stars', 0)) for user in user_data]
    products = [Product(item['item_id'], item.get('category', ''), item['stars'], item['review_count']) for item in item_data]

    simulation = Simulation(users, products, user_data, item_data)
    simulation.run()
    simulation.evaluate()
    simulation.visualize()
    simulation.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()