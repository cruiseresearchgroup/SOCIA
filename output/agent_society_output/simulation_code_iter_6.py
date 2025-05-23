import os
import json
import random
import logging
import openai
from openai import APIError, RateLimitError, Timeout, APIConnectionError, OpenAIError
from openai import OpenAI
import pandas as pd
import time
from typing import List, Dict, Tuple, Any, Union
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
        logging.error("OpenAI API key is missing. Please provide the API key in keys.py or as an environment variable OPENAI_API_KEY.")
        # sys.exit(1) # Removed to allow further inspection even if key is missing initially

# Initialize OpenAI client
client = OpenAI(api_key=openai.api_key) # Initialize client with the key

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
        # Adapt to review_history possibly containing dicts or tuples
        historical_review_texts = []
        for rev in self.review_history:
            if isinstance(rev, dict):
                if rev.get('item_id') == product.id:
                    historical_review_texts.append(rev.get('text', '')) # Assuming 'text' key for review content
            elif isinstance(rev, tuple) and len(rev) >= 3:
                if rev[0] == product.id:
                    historical_review_texts.append(rev[2])
            
        historical_reviews_concat = ' '.join(filter(None, historical_review_texts))
        
        prompt = (f"User {self.id} with preferences {self.preferences} is reviewing product {product.id} "
                  f"in category {product.category}. Consider historical reviews: {historical_reviews_concat}")

        response = None
        for attempt in range(3):
            try:
                response = client.completions.create( # Use client.completions.create
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=100
                )
                if response.choices: # Check response.choices
                    break
            except APIError as e: # Catch specific API errors
                logging.error(f"OpenAI API Error generating review text: {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
            except OpenAIError as e: # Catch other OpenAI errors
                logging.error(f"OpenAI Error generating review text: {e}")
                # Decide if retry is appropriate for non-API specific OpenAIError
                break # For now, break on general OpenAIError
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error generating review text: {e}")
                break
                            
        review_text = response.choices[0].text.strip() if response and response.choices else "Error generating review text." # Access text via response.choices[0].text
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
    
    def __init__(self, user_data: List[Dict[str, Any]], item_data: List[Dict[str, Any]], review_data: Union[Dict[str, Any], List[Dict[str, Any]]]):
        # Convert user_data list to a dictionary keyed by user_id
        self.user_data = {user['user_id']: user for user in user_data}
        # Convert item_data list to a dictionary keyed by item_id
        self.item_data = {item['item_id']: item for item in item_data}
        
        # Process review_data into the expected nested dictionary structure
        self.review_data: Dict[str, Dict[str, List[Any]]] = {}
        if isinstance(review_data, list):
            for review in review_data:
                u_id = review.get('user_id')
                i_id = review.get('item_id')
                if u_id and i_id:
                    if u_id not in self.review_data:
                        self.review_data[u_id] = {}
                    if i_id not in self.review_data[u_id]:
                        self.review_data[u_id][i_id] = []
                    # Assuming the original review_sample.json might contain more than just (user_id, item_id, rating, text)
                    # For now, we just store the whole review dict. Adjust if only specific fields are needed.
                    self.review_data[u_id][i_id].append(review) 
        elif isinstance(review_data, dict):
            # If it's already a dict, assume it's in the correct nested structure or can be used as is.
            # This part might need refinement if the pre-loaded dict structure is different from the target.
            self.review_data = review_data
        else:
            logging.warning(f"Unexpected type for review_data: {type(review_data)}. Expected Dict or List.")

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
                response = client.completions.create( # Use client.completions.create
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150
                )
                if response.choices: # Check response.choices
                    break
            except APIError as e: # Catch specific API errors
                logging.error(f"OpenAI API Error during reasoning: {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
            except OpenAIError as e: # Catch other OpenAI errors
                logging.error(f"OpenAI Error during reasoning: {e}")
                break # For now, break on general OpenAIError
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error during reasoning: {e}")
                break
                    
        review_text = response.choices[0].text.strip() if response and response.choices else "Error generating review text." # Access text via response.choices[0].text
        # Logic to extract rating is missing here. For now, just return a placeholder.
        # Placeholder for rating extraction logic. This needs to be implemented.
        rating = user.rate_product(product) # Restored original rating logic
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
            logging.info(f"Average rating across all products: {average_rating:.2f}")
        else:
            logging.info("No reviews available for evaluation.")

    def save_results(self, filename: str) -> None:
        """Save the product results of the simulation to a CSV file."""
        product_results = []
        for p in self.products:
            product_results.append({
                "product_id": p.id,
                "category": p.category,
                "final_average_rating": p.average_rating,
                "final_review_count": p.review_count
            })
        
        if product_results:
            df = pd.DataFrame(product_results)
            df.to_csv(filename, index=False)
            logging.info(f"Product results saved to {filename}")
        else:
            logging.info("No product results to save.")

def main() -> None:
    """Main function to initialize entities and run the simulation."""
    if not user_data or not item_data:
        logging.error("User data or item data not loaded. Exiting.")
        return

    users = [User(user['user_id'], user.get('preferences', {}), [], user.get('rating_tendency', 0)) for user in user_data]
    products = [Product(item['item_id'], item.get('category', ''), item.get('stars', 0.0), item.get('review_count', 0), item.get('attributes', {})) for item in item_data]

    simulation = Simulation(users, products, user_data, item_data, review_data)
    simulation.run()
    simulation.evaluate()
    simulation.save_results("results.csv")

# Execute main for both direct execution and sandbox wrapper invocation
main()