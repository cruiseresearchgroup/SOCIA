import os
import json
import random
import logging
import pandas as pd
import openai
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/agent_society/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    try:
        from keys import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY
    except ImportError:
        logging.error("OpenAI API key is missing. Please provide the API key through 'keys.py' or as an environment variable.")
        raise RuntimeError("OpenAI API key is required for running the simulation.")

def load_json_file(filename: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found in {DATA_DIR}.")
    except json.JSONDecodeError:
        logging.error(f"The file {filename} contains malformed JSON.")
    except PermissionError:
        logging.error(f"Permission denied when trying to open the file {filename}.")
    except IOError as e:
        logging.error(f"An unexpected IO error occurred while reading the file {filename}: {e}")
    return {}

user_data = load_json_file('user_sample.json')
item_data = load_json_file('item_sample.json')
review_data = load_json_file('review_sample.json')
amazon_data = load_json_file('amazon_train_sample.json')
goodreads_data = load_json_file('goodreads_train_sample.json')
yelp_data = load_json_file('yelp_train_sample.json')

class User:
    def __init__(self, user_id: str, preferences: Dict[str, Any], review_history: List[Tuple[str, int, str]], rating_tendency: float):
        """User entity in the simulation."""
        self.id = user_id
        self.preferences = preferences
        self.review_history = review_history
        self.rating_tendency = rating_tendency

    def write_review(self, product: 'Product') -> Tuple[int, str]:
        """Simulates writing a review for a product."""
        review_text = self.generate_review_text(product)
        rating = self.rate_product(product)
        self.review_history.append((product.id, rating, review_text))
        return rating, review_text

    def generate_review_text(self, product: 'Product') -> str:
        """Generate review text using OpenAI's API."""
        historical_reviews_text = ' '.join(rev[2] for rev in self.review_history if rev[0] == product.id)
        prompt = (f"User {self.id} with preferences {self.preferences} is reviewing product {product.id} "
                  f"in category {product.category}. Consider historical reviews: {historical_reviews_text}")
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
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
        preference_score = self.preferences.get(product.category, 0)
        rating = base_rating + preference_score * self.rating_tendency
        return max(1, min(5, round(rating)))

class Product:
    def __init__(self, item_id: str, category: str, average_rating: float, review_count: int, attributes: Dict[str, Any]):
        """Product entity in the simulation."""
        self.id = item_id
        self.category = category
        self.average_rating = average_rating
        self.review_count = review_count
        self.attributes = attributes

    def receive_review(self, rating: int) -> None:
        """Process an incoming review by updating the review count and average rating."""
        self.review_count += 1
        self.average_rating = ((self.average_rating * (self.review_count - 1)) + rating) / self.review_count

    def receive_rating(self, rating: int) -> None:
        """Update the product rating."""
        self.receive_review(rating)

class PlanningAgent:
    def create_plan(self, user_id: str, item_id: str) -> Dict[str, str]:
        """Create a plan for the user to write a review for an item."""
        return {"user_id": user_id, "item_id": item_id, "action": "write_review"}

class MemoryAgent:
    def __init__(self, user_data: Dict[str, Any], item_data: Dict[str, Any], review_data: Dict[str, Any]):
        """Memory agent to store and retrieve user, item, and historical review information."""
        self.user_data = user_data
        self.item_data = item_data
        self.review_data = review_data

    def retrieve_user_info(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user information based on user ID."""
        user_info = self.user_data.get(user_id, {})
        if not user_info:
            logging.warning(f"User {user_id} not found.")
        return user_info

    def retrieve_item_info(self, item_id: str) -> Dict[str, Any]:
        """Retrieve item information based on item ID."""
        item_info = self.item_data.get(item_id, {})
        if not item_info:
            logging.warning(f"Item {item_id} not found.")
        return item_info

    def retrieve_review_history(self, user_id: str, item_id: str) -> List[Tuple[str, int, str]]:
        """Retrieve historical review data for a user and item."""
        return self.review_data.get(user_id, {}).get(item_id, [])

class ReasoningAgent:
    def simulate_review(self, user: User, product: Product) -> Tuple[int, str]:
        """Simulate a review process by a user for a product using LLM."""
        prompt = (f"Simulate a review by user {user.id} for product {product.id}. "
                  f"User preferences: {user.preferences}. Product attributes: {product.attributes}.")
        try:
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=150
            )
            review_text = response['choices'][0]['text'].strip()
            rating = user.rate_product(product)
        except (openai.error.OpenAIError, openai.error.RateLimitError, openai.error.Timeout) as e:
            logging.error(f"Error during reasoning: {e}")
            rating, review_text = user.rate_product(product), "Error generating review text."
        return rating, review_text

class Simulation:
    """Main simulation class coordinating all agents and entities."""
    def __init__(self, users: List[User], products: List[Product], user_data: Dict[str, Any], item_data: Dict[str, Any], review_data: Dict[str, Any]):
        self.users = users
        self.products = products
        self.planning_agent = PlanningAgent()
        self.memory_agent = MemoryAgent(user_data, item_data, review_data)
        self.reasoning_agent = ReasoningAgent()

    def run(self, days: int = 30) -> None:
        """Run the simulation for a given number of days."""
        for day in range(days):
            for user in self.users:
                if random.random() < 0.1:
                    product = self.select_product_for_user(user)
                    plan = self.planning_agent.create_plan(user.id, product.id)
                    user_info = self.memory_agent.retrieve_user_info(plan['user_id'])
                    item_info = self.memory_agent.retrieve_item_info(plan['item_id'])
                    historical_reviews = self.memory_agent.retrieve_review_history(plan['user_id'], plan['item_id'])
                    if user_info and item_info:
                        user.review_history = historical_reviews
                        rating, review = self.reasoning_agent.simulate_review(user, product)
                        product.receive_review(rating)

    def select_product_for_user(self, user: User) -> Product:
        """Select a product for the user to review based on their preferences and history."""
        preferred_categories = [category for category, score in user.preferences.items() if score > 0]
        preferred_products = [product for product in self.products if product.category in preferred_categories]
        if preferred_products:
            return random.choice(preferred_products)
        return random.choice(self.products)

    def evaluate(self) -> None:
        """Evaluate the results of the simulation."""
        total_review_count = 0
        total_rating_sum = 0
        for product in self.products:
            total_review_count += product.review_count
            total_rating_sum += product.average_rating * product.review_count
        if total_review_count > 0:
            average_rating = total_rating_sum / total_review_count
            logging.info(f"Average rating across all products: {average_rating}")
        else:
            logging.info("No reviews available for evaluation.")

    def visualize(self) -> None:
        """Visualize the results of the simulation."""
        product_ids = [product.id for product in self.products]
        average_ratings = [product.average_rating for product in self.products]
        review_counts = [product.review_count for product in self.products]

        plt.figure(figsize=(12, 6))
        plt.bar(product_ids, average_ratings, color='skyblue')
        plt.xlabel('Product ID')
        plt.ylabel('Average Rating')
        plt.title('Average Rating per Product')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.bar(product_ids, review_counts, color='lightgreen')
        plt.xlabel('Product ID')
        plt.ylabel('Review Count')
        plt.title('Review Count per Product')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

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

    users = [User(user['user_id'], user.get('preferences', {}), [], user.get('rating_tendency', 0)) for user in user_data]
    products = [Product(item['item_id'], item.get('category', ''), item['average_rating'], item['review_count'], item.get('attributes', {})) for item in item_data]

    simulation = Simulation(users, products, user_data, item_data, review_data)
    simulation.run()
    simulation.evaluate()
    simulation.visualize()
    simulation.save_results("results.csv")


# Execute main for both direct execution and sandbox wrapper invocation
main()