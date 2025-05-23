import os
import json
import random
import logging
import openai
import pandas as pd
import time
from typing import List, Dict, Tuple, Any
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

# Configure paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/agent_society/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)

# Add the directory containing keys.py to sys.path
keys_dir = DATA_DIR
if keys_dir not in sys.path:
    sys.path.insert(0, keys_dir)

# Load OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')
if openai.api_key is None:
    try:
        from keys import OPENAI_API_KEY
        openai.api_key = OPENAI_API_KEY
    except ImportError:
        logging.error("OpenAI API key is missing. Please provide the API key through 'keys.py' or as an environment variable.")
        # Try to load from the absolute path if direct import fails
        try:
            keys_abs_path = os.path.join(keys_dir, "keys.py")
            if os.path.exists(keys_abs_path):
                logging.info(f"Attempting to load keys.py from absolute path: {keys_abs_path}")
                # This part is tricky as direct execution of an arbitrary file is a security risk.
                # A better way would be to ensure keys.py is a proper module.
                # For now, we still rely on it being in sys.path and being importable.
                # If the sys.path modification above works, this fallback might not be strictly necessary
                # or would need a more robust way to load a variable from a specific file path.
                # Re-raising the original error if this path also doesn't lead to a successful import.
                from keys import OPENAI_API_KEY
                openai.api_key = OPENAI_API_KEY
            else:
                logging.error(f"keys.py not found at expected location: {keys_abs_path}")
                raise RuntimeError("OpenAI API key is required for running the simulation and keys.py not found.")

        except ImportError: # Catching ImportError again if the second attempt also fails
            logging.error("Failed to import OPENAI_API_KEY from keys.py even after path adjustments.")
            raise RuntimeError("OpenAI API key is required for running the simulation.")
    except Exception as e: # Catch any other unexpected error during key loading
        logging.error(f"An unexpected error occurred while loading OpenAI API key: {e}")
        raise RuntimeError("OpenAI API key is required for running the simulation.")

def load_json_file(filename: str) -> Dict[str, Any]:
    """Load data from a JSON file."""
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            if not data:
                logging.warning(f"The file {filename} is empty.")
            return data
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found in {DATA_DIR}.")
    except json.JSONDecodeError:
        logging.error(f"The file {filename} contains malformed JSON.")
    except PermissionError:
        logging.error(f"Permission denied when trying to open the file {filename}.")
    except IOError as e:
        logging.error(f"An unexpected IO error occurred while reading the file {filename}: {e}")
    return {}

# Load data
user_data = load_json_file('user_sample.json')
item_data = load_json_file('item_sample.json')
review_data = load_json_file('review_sample.json')
amazon_data = load_json_file('amazon_train_sample.json')
goodreads_data = load_json_file('goodreads_train_sample.json')
yelp_data = load_json_file('yelp_train_sample.json')

class User:
    """User entity in the simulation."""
    
    def __init__(self, user_id: str, preferences: Dict[str, Any], review_history: List[Tuple[str, int, str]], rating_tendency: float):
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

        response = None
        for attempt in range(3):
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100
                )
                if 'choices' in response and response['choices']:
                    break
            except (openai.error.OpenAIError, openai.error.APIConnectionError, openai.error.Timeout, openai.error.RateLimitError) as e:
                logging.error(f"Error generating review text: {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
                    
        review_text = response['choices'][0]['text'].strip() if response and 'choices' in response and response['choices'] else "Error generating review text."
        return review_text

    def rate_product(self, product: 'Product') -> int:
        """Rate the product based on user preferences."""
        base_rating = random.uniform(1, 5)
        preference_score = self.preferences.get(product.category, 0)
        rating = base_rating + preference_score * self.rating_tendency
        return max(1, min(5, round(rating)))

class Product:
    """Product entity in the simulation."""
    
    def __init__(self, item_id: str, category: str, average_rating: float, review_count: int, attributes: Dict[str, Any]):
        self.id = item_id
        self.category = category
        self.average_rating = average_rating
        self.review_count = review_count
        self.attributes = attributes

    def receive_review(self, rating: int) -> None:
        """Process an incoming review by updating the review count and average rating."""
        self.review_count += 1
        self.average_rating = ((self.average_rating * (self.review_count - 1)) + rating) / self.review_count

class PlanningAgent:
    """Agent responsible for creating plans for users to review items."""
    
    def create_plan(self, user_id: str, item_id: str) -> Dict[str, str]:
        """Create a plan for a user to review an item."""
        return {"user_id": user_id, "item_id": item_id, "action": "write_review"}

class MemoryAgent:
    """Memory agent to store and retrieve user, item, and historical review information."""
    
    def __init__(self, user_data: Dict[str, Any], item_data: Dict[str, Any], review_data: Dict[str, Any]):
        self.user_data = user_data
        self.item_data = item_data
        self.review_data = review_data

    def retrieve_user_info(self, user_id: str) -> Dict[str, Any]:
        """Retrieve user information."""
        user_info = self.user_data.get(user_id, {})
        if not user_info:
            logging.warning(f"User {user_id} not found.")
        return user_info

    def retrieve_item_info(self, item_id: str) -> Dict[str, Any]:
        """Retrieve item information."""
        item_info = self.item_data.get(item_id, {})
        if not item_info:
            logging.warning(f"Item {item_id} not found.")
        return item_info

    def retrieve_review_history(self, user_id: str, item_id: str) -> List[Tuple[str, int, str]]:
        """Retrieve historical review information."""
        return self.review_data.get(user_id, {}).get(item_id, [])

class ReasoningAgent:
    """Reasoning agent using LLM to simulate the review process."""
    
    def simulate_review(self, user: User, product: Product, platform: str) -> Tuple[int, str]:
        """Simulate a review by a user for a product."""
        prompt = (f"Simulate a review by user {user.id} for product {product.id} on {platform}. "
                  f"User preferences: {user.preferences}. Product attributes: {product.attributes}.")
        
        response = None
        for attempt in range(3):
            try:
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150
                )
                if 'choices' in response and response['choices']:
                    break
            except (openai.error.OpenAIError, openai.error.APIConnectionError, openai.error.Timeout, openai.error.RateLimitError) as e:
                logging.error(f"Error during reasoning: {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
                    
        review_text = response['choices'][0]['text'].strip() if response and 'choices' in response and response['choices'] else "Error generating review text."
        rating = user.rate_product(product)
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
                try:
                    if random.random() < 0.1:
                        product = self.select_product_for_user(user)
                        plan = self.planning_agent.create_plan(user.id, product.id)
                        user_info = self.memory_agent.retrieve_user_info(plan['user_id'])
                        item_info = self.memory_agent.retrieve_item_info(plan['item_id'])
                        historical_reviews = self.memory_agent.retrieve_review_history(plan['user_id'], plan['item_id'])
                        if user_info and item_info:
                            user.review_history = historical_reviews
                            platform = random.choice(['Amazon', 'Goodreads', 'Yelp'])
                            rating, review = self.reasoning_agent.simulate_review(user, product, platform)
                            product.receive_review(rating)
                except Exception as e:
                    logging.error(f"An error occurred during the simulation run: {e}")

    def select_product_for_user(self, user: User) -> Product:
        """Select a product for the user to review based on their preferences and history."""
        try:
            preferred_categories = [category for category, score in user.preferences.items() if score > 0]
            preferred_products = [product for product in self.products if product.category in preferred_categories]
            if preferred_products:
                return random.choice(preferred_products)
            return random.choice(self.products)
        except Exception as e:
            logging.error(f"Error selecting product for user {user.id}: {e}")
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

    def save_results(self, filename: str) -> None:
        """Save the results of the simulation to a CSV file."""
        results = {"users": [user.id for user in self.users],
                   "products": [product.id for product in self.products]}
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

def main() -> None:
    """Main function to initialize entities and run the simulation."""
    if not user_data or not item_data:
        logging.error("User data or item data not loaded. Exiting.")
        return

    users = [User(user['user_id'], user.get('preferences', {}), [], user.get('rating_tendency', 0)) for user in user_data]
    products = [Product(item['item_id'], item.get('category', ''), item['average_rating'], item['review_count'], item.get('attributes', {})) for item in item_data]

    simulation = Simulation(users, products, user_data, item_data, review_data)
    simulation.run()
    simulation.evaluate()
    simulation.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()