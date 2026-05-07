import os
import json
import random
import openai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Any

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
    """
    Represents a user in the simulation with specific preferences and review history.

    :param user_id: Unique identifier for the user
    :param preferences: Dictionary containing user preferences
    :param review_history: List of user's past reviews
    """
    def __init__(self, user_id: str, preferences: Dict[str, Any], review_history: List[str]):
        self.user_id = user_id
        self.preferences = preferences
        self.review_history = review_history

    def write_review(self, product, llm_agent) -> str:
        """
        Simulates writing a review by invoking the LLM agent to generate text based on preferences and product attributes.
        
        :param product: The product to be reviewed
        :param llm_agent: The LLM agent responsible for generating the review
        :return: Generated review text
        """
        review = llm_agent.reason_review(self, product)
        product.receive_review(review, None)
        return review

    def give_star_rating(self, product, llm_agent) -> int:
        """
        Simulates giving a star rating by invoking the LLM agent to predict a rating based on user preferences.
        
        :param product: The product to be rated
        :param llm_agent: The LLM agent responsible for predicting the rating
        :return: Predicted star rating
        """
        rating = llm_agent.reason_star_rating(self, product)
        product.receive_review(None, rating)
        return rating

class Product:
    """
    Represents a product with attributes relevant to user interaction.

    :param product_id: Unique identifier for the product
    :param category: Category of the product
    :param average_rating: Average rating of the product
    """
    def __init__(self, product_id: str, category: str, average_rating: float):
        self.product_id = product_id
        self.category = category
        self.average_rating = average_rating
        self.reviews = []
        self.rating_sum = 0
        self.rating_count = 0

    def receive_review(self, review_text: Union[str, None], star_rating: Union[int, None]):
        """
        Updates the product's attributes based on new reviews and star ratings.
        
        :param review_text: Text of the review
        :param star_rating: Star rating given to the product
        """
        if review_text:
            self.reviews.append(review_text)
        if star_rating is not None:
            self.rating_sum += star_rating
            self.rating_count += 1
            self.average_rating = self.rating_sum / self.rating_count if self.rating_count else self.average_rating

class LLMAgent:
    """
    Represents the LLM agent responsible for reasoning and generating reviews and ratings.

    :param openai_key: API key for accessing the OpenAI services
    """
    def __init__(self, openai_key: str):
        try:
            openai.api_key = openai_key
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI API: {e}")

    def reason_review(self, user: User, product: Product) -> str:
        """
        Uses LLM to generate a review based on user and product data.
        
        :param user: The user writing the review
        :param product: The product being reviewed
        :return: Generated review text
        """
        # Placeholder for LLM API call
        return f"Review for {product.product_id} by {user.user_id}"

    def reason_star_rating(self, user: User, product: Product) -> int:
        """
        Uses LLM to predict a star rating based on user and product data.
        
        :param user: The user giving the rating
        :param product: The product being rated
        :return: Predicted star rating
        """
        # Placeholder for LLM API call
        return random.randint(1, 5)

class SimulationEnvironment:
    """
    Sets up and manages the simulation environment and interactions between entities.
    """
    def __init__(self, num_users: int, num_products: int):
        self.users = []
        self.products = []
        self.llm_agent = LLMAgent(openai_key=os.environ["OPENAI_API_KEY"])
        self.load_entities(num_users, num_products)

    def load_entities(self, num_users: int, num_products: int):
        """
        Loads users and products from data files.
        
        :param num_users: Number of users to load
        :param num_products: Number of products to load
        """
        try:
            with open(user_data_file, 'r') as f:
                user_data = json.load(f)
                for user_id, data in user_data.items():
                    if len(self.users) >= num_users:
                        break
                    user = User(user_id, data['preferences'], data['review_history'])
                    self.users.append(user)

            with open(item_data_file, 'r') as f:
                product_data = json.load(f)
                for product_id, data in product_data.items():
                    if len(self.products) >= num_products:
                        break
                    product = Product(product_id, data['category'], data['average_rating'])
                    self.products.append(product)
        except FileNotFoundError:
            print("Error loading data: File not found.")
        except PermissionError:
            print("Error loading data: Permission denied.")
        except json.JSONDecodeError:
            print("Error loading data: JSON decoding error.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def run_simulation(self):
        """
        Executes the main simulation loop.
        """
        for user in self.users:
            products_to_review = random.sample(self.products, min(len(self.products), 5))  # Review up to 5 products
            for product in products_to_review:
                user.write_review(product, self.llm_agent)
                user.give_star_rating(product, self.llm_agent)

    def visualize(self):
        """
        Visualizes simulation results.
        """
        ratings = [product.average_rating for product in self.products]
        plt.hist(ratings, bins=5, range=(1, 5), edgecolor='black')
        plt.title('Distribution of Average Ratings')
        plt.xlabel('Average Rating')
        plt.ylabel('Number of Products')
        plt.show()

    def save_results(self, filename: str):
        """
        Saves simulation results to a file.
        
        :param filename: Name of the file to save results to
        """
        results = {
            'product_id': [product.product_id for product in self.products],
            'average_rating': [product.average_rating for product in self.products]
        }
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)

def main():
    """
    Main function to initialize and run the simulation, visualize and save results.
    """
    simulation = SimulationEnvironment(num_users=100, num_products=50)
    simulation.run_simulation()
    simulation.visualize()
    simulation.save_results("results.csv")

main()