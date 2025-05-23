import os
import json
import random
import logging
import openai
from openai import APIError, RateLimitError, Timeout, APIConnectionError, OpenAIError
from openai import OpenAI
import pandas as pd
import time
from typing import List, Dict, Tuple, Any, Union, Optional
import sys
from sklearn.metrics import mean_absolute_error
from rouge_score import rouge_scorer # For ROUGE score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure paths
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
DATA_PATH = os.environ.get("DATA_PATH", "data_fitting/agent_society/")
DATA_DIR = os.path.join(PROJECT_ROOT, DATA_PATH)
# Change output directory to be in the same folder as the script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add the directory containing keys.py to sys.path
keys_dir = DATA_DIR
if keys_dir not in sys.path:
    sys.path.insert(0, keys_dir)

# Load OpenAI API key
openai_api_key_loaded = False
try:
    from keys import OPENAI_API_KEY
    openai.api_key = OPENAI_API_KEY
    openai_api_key_loaded = True
except ImportError:
    logging.warning("Could not import OPENAI_API_KEY from keys.py. Checking environment variable.")
    openai.api_key = os.getenv('OPENAI_API_KEY')
    if openai.api_key:
        openai_api_key_loaded = True

if not openai_api_key_loaded:
    logging.error("OpenAI API key is missing. Please provide the API key in keys.py or as an environment variable OPENAI_API_KEY.")
    # For evaluation, we might want to proceed if user/item data is still useful, 
    # but LLM calls will fail.
    # Consider exiting if LLM calls are essential for the script's purpose.
    # sys.exit(1) 

# Initialize OpenAI client
# Ensure API key is not None before initializing client, or handle potential error
if openai.api_key:
    client = OpenAI(api_key=openai.api_key)
else:
    client = None
    logging.warning("OpenAI client not initialized as API key is missing. LLM calls will fail.")

def load_json_file(filename: str, is_evaluation_sample: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Load data from a JSON file.
    If is_evaluation_sample is True, assumes a structure like {key: record, key: record} and returns a list of records.
    Otherwise, loads as a general dictionary or list based on file content.
    """
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file: # Added encoding
            data = json.load(file)
            if not data:
                logging.warning(f"The file {filename} is empty.")
                return [] if is_evaluation_sample else {}
            
            if is_evaluation_sample and isinstance(data, dict):
                return list(data.values()) # Extract records from the dictionary
            return data
    except FileNotFoundError:
        logging.error(f"The file {filename} was not found in {DATA_DIR}.")
    except json.JSONDecodeError:
        logging.error(f"The file {filename} contains malformed JSON.")
    except PermissionError:
        logging.error(f"Permission denied when trying to open the file {filename}.")
    except IOError as e:
        logging.error(f"An unexpected IO error occurred while reading the file {filename}: {e}")
    return [] if is_evaluation_sample else {}

# Load data
user_data = load_json_file('user_sample.json')
item_data = load_json_file('item_sample.json')
review_data = load_json_file('review_sample.json')
amazon_data = load_json_file('amazon_train_sample.json')
goodreads_data = load_json_file('goodreads_train_sample.json')
yelp_data = load_json_file('yelp_train_sample.json')

def load_review_history(review_data: List[Dict[str, Any]], user_id: str = None, item_id: str = None) -> List[Dict[str, Any]]:
    """
    Load review history for a specific user or item.
    If user_id is provided, returns that user's review history.
    If item_id is provided, returns reviews for that item.
    If both are provided, returns reviews by that user for that item.
    """
    filtered_reviews = []
    for review in review_data:
        if user_id and item_id:
            if review.get('user_id') == user_id and review.get('item_id') == item_id:
                filtered_reviews.append(review)
        elif user_id:
            if review.get('user_id') == user_id:
                filtered_reviews.append(review)
        elif item_id:
            if review.get('item_id') == item_id:
                filtered_reviews.append(review)
    return filtered_reviews

class User:
    """User entity in the simulation."""
    
    def __init__(self, user_id: str, preferences: Optional[Dict[str, Any]] = None, 
                 review_history: Optional[List[Dict[str, Any]]] = None, 
                 rating_tendency: float = 0.0):
        self.id = user_id
        self.preferences = preferences if preferences is not None else {}
        self.review_history = review_history if review_history is not None else []
        self.rating_tendency = rating_tendency

    def _format_review_history_for_prompt(self) -> str:
        """Format review history for prompt in a concise way."""
        if not self.review_history:
            return "No previous reviews available."
        
        formatted_reviews = []
        for review in self.review_history[:5]:  # Limit to last 5 reviews
            formatted_reviews.append(
                f"- Rating: {review.get('stars', 'N/A')} stars, "
                f"Review: \"{review.get('review', 'N/A')}\""
            )
        return "\nUser's recent reviews:\n" + "\n".join(formatted_reviews)

    def generate_review_text(self, product: 'Product', client_instance: Optional[OpenAI] = None) -> str:
        """Generate review text using OpenAI's API."""
        if not client_instance:
            logging.error("OpenAI client not provided to generate_review_text.")
            return "Error: OpenAI client not available."

        # Format user's review history
        user_review_history = self._format_review_history_for_prompt()
        
        # Format product's review history
        product_review_history = product.get_formatted_review_history()
        
        prompt = (
            f"Task: Generate a product review based on the following context:\n\n"
            f"1. User Information:\n"
            f"- User ID: {self.id}\n"
            f"- User Preferences: {self.preferences}\n"
            f"- User Rating Tendency: {self.rating_tendency*5:.1f} stars average\n"
            f"{user_review_history}\n\n"
            f"2. Product Information:\n"
            f"- Product ID: {product.id}\n"
            f"- Category: {product.category}\n"
            f"- Attributes: {product.attributes}\n"
            f"- Current Average Rating: {product.average_rating:.1f} stars\n"
            f"{product_review_history}\n\n"
            f"Please write a detailed and authentic product review that reflects this user's writing style "
            f"and takes into account their preferences and rating tendency, as well as the product's characteristics "
            f"and review history. The review should be genuine and specific to this product."
        )

        response_content = None
        for attempt in range(3):
            try:
                response = client_instance.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=250  # Increased token limit for more detailed reviews
                )
                if response.choices:
                    response_content = response.choices[0].message.content.strip()
                    break
            except APIError as e:
                logging.error(f"OpenAI API Error (User: {self.id}, Product: {product.id}): {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call for review text. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
            except OpenAIError as e:
                logging.error(f"General OpenAI Error (User: {self.id}, Product: {product.id}): {e}")
                break 
            except Exception as e:
                logging.error(f"Unexpected error generating review text (User: {self.id}, Product: {product.id}): {e}")
                break
                            
        return response_content if response_content else "Error: Could not generate review text after multiple attempts."

    def rate_product(self, product: 'Product', client_instance: Optional[OpenAI] = None, generated_review_text: Optional[str] = None) -> int:
        """Rate the product based on user history, preferences, and generated review."""
        if not client_instance:
            logging.error("OpenAI client not provided to rate_product.")
            return 1

        # Format user's review history
        user_review_history = self._format_review_history_for_prompt()
        
        # Format product's review history
        product_review_history = product.get_formatted_review_history()

        prompt_for_rating = (
            f"Task: Predict a star rating (1-5) based on the following context:\n\n"
            f"1. User Profile:\n"
            f"- User ID: {self.id}\n"
            f"- Preferences: {self.preferences}\n"
            f"- Rating Tendency: {self.rating_tendency*5:.1f} stars average\n"
            f"{user_review_history}\n\n"
            f"2. Product Information:\n"
            f"- Product ID: {product.id}\n"
            f"- Category: {product.category}\n"
            f"- Attributes: {product.attributes}\n"
            f"- Current Average Rating: {product.average_rating:.1f} stars\n"
            f"{product_review_history}\n\n"
            f"3. Generated Review:\n"
            f"\"{generated_review_text if generated_review_text else 'No review text available.'}\"\n\n"
            f"Based on all this information, predict the star rating (an integer between 1 and 5) "
            f"this user would give. Consider the user's rating history and tendencies, "
            f"as well as the product's characteristics and review history. "
            f"Return only the integer rating."
        )
        
        response_content = None
        for attempt in range(3):
            try:
                response = client_instance.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt_for_rating}],
                    max_tokens=10
                )
                if response.choices:
                    raw_rating_text = response.choices[0].message.content.strip()
                    # Try to extract an integer
                    import re
                    match = re.search(r'\b([1-5])\b', raw_rating_text)
                    if match:
                        response_content = int(match.group(1))
                        break
                    else:
                        logging.warning(f"Could not parse rating from LLM response: '{raw_rating_text}'. User: {self.id}, Product: {product.id}")
            except APIError as e:
                logging.error(f"OpenAI API Error getting rating (User: {self.id}, Product: {product.id}): {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call for rating. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
            except OpenAIError as e:
                logging.error(f"General OpenAI Error getting rating (User: {self.id}, Product: {product.id}): {e}")
                break
            except Exception as e:
                logging.error(f"Unexpected error getting rating (User: {self.id}, Product: {product.id}): {e}")
                break
        
        if response_content is None or not (1 <= response_content <= 5):
            logging.warning(f"Failed to get valid rating from LLM for User: {self.id}, Product: {product.id}. Using fallback.")
            # Enhanced fallback using user's rating tendency and product average
            base_rating = (self.rating_tendency * 5 + product.average_rating) / 2
            preference_score = self.preferences.get(product.category, 0)
            rating = base_rating + preference_score
            return max(1, min(5, round(rating)))
            
        return response_content

class Product:
    """Product entity in the simulation."""
    
    def __init__(self, item_id: str, category: Optional[str] = "Unknown", 
                 average_rating: float = 0.0, review_count: int = 0, 
                 attributes: Optional[Dict[str, Any]] = None,
                 review_history: Optional[List[Dict[str, Any]]] = None):
        self.id = item_id
        self.category = category
        self.average_rating = average_rating
        self.review_count = review_count
        self.attributes = attributes if attributes is not None else {}
        self.review_history = review_history if review_history is not None else []

    def get_formatted_review_history(self) -> str:
        """Format review history for prompt in a concise way."""
        if not self.review_history:
            return "No previous reviews available for this product."
        
        # Calculate rating distribution
        rating_dist = {i: 0 for i in range(1, 6)}
        for review in self.review_history:
            stars = review.get('stars')
            if isinstance(stars, (int, float)) and 1 <= stars <= 5:
                rating_dist[int(stars)] += 1
        
        # Format rating distribution
        dist_str = "Rating distribution:\n" + "\n".join(
            f"{stars} stars: {count} reviews" 
            for stars, count in rating_dist.items()
        )
        
        # Add sample reviews (most recent 3)
        recent_reviews = self.review_history[-3:]  # Get last 3 reviews
        sample_reviews = "\nRecent reviews:\n" + "\n".join(
            f"- Rating: {review.get('stars', 'N/A')} stars, "
            f"Review: \"{review.get('review', 'N/A')}\""
            for review in recent_reviews
        )
        
        return f"\n{dist_str}\n{sample_reviews}"

class ReasoningAgent:
    """Reasoning agent using LLM to simulate the review process for evaluation."""
    
    def __init__(self, client_instance: Optional[OpenAI]):
        self.client = client_instance

    def generate_review_and_rating(self, user: User, product: Product, platform_context: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Simulates generating a review text and then predicting a star rating based on that text.
        This method is primarily for the evaluation flow.
        Args:
            platform_context: e.g., "Amazon", "Yelp" - to give context to the LLM.
        """
        if not self.client:
            logging.error("ReasoningAgent: OpenAI client not initialized.")
            return None, None

        # 1. Generate review text
        generated_review_text = user.generate_review_text(product, client_instance=self.client)
        if "Error:" in generated_review_text: # Check for error message
             logging.warning(f"Failed to generate review text for User: {user.id}, Product: {product.id}")
             # Optionally return None, None or attempt rating with no text
        
        # 2. Predict star rating based on the generated review text
        predicted_star_rating = user.rate_product(product, client_instance=self.client, generated_review_text=generated_review_text)
        
        return predicted_star_rating, generated_review_text

def evaluate_on_samples(
    reasoning_agent: ReasoningAgent,
    user_master_data: Dict[str, Dict[str, Any]],
    item_master_data: Dict[str, Dict[str, Any]],
    review_data: List[Dict[str, Any]]
):
    """
    Loads samples from specified JSON files, performs inference, compares with ground truth,
    and saves metrics.
    """
    if not client:
        logging.error("OpenAI client is not initialized. Cannot proceed with evaluation.")
        return

    source_files = ['amazon_train_sample.json', 'goodreads_train_sample.json', 'yelp_train_sample.json']
    num_samples_per_file = 10
    all_evaluation_results = []
    
    # Initialize ROUGE scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for source_file in source_files:
        logging.info(f"Processing samples from: {source_file}")
        evaluation_data = load_json_file(source_file, is_evaluation_sample=True)
        if not evaluation_data:
            logging.warning(f"No data loaded from {source_file}. Skipping.")
            continue

        if len(evaluation_data) < num_samples_per_file:
            logging.warning(f"File {source_file} has less than {num_samples_per_file} records. Using all {len(evaluation_data)} records.")
            sampled_records = evaluation_data
        else:
            sampled_records = random.sample(evaluation_data, num_samples_per_file)
        
        file_star_maes = []

        for record in sampled_records:
            user_id = record.get('user_id')
            item_id = record.get('item_id')
            gt_stars = record.get('stars')
            gt_review = record.get('review', "")

            if user_id is None or item_id is None or gt_stars is None:
                logging.warning(f"Skipping record due to missing user_id, item_id, or stars: {record}")
                continue
            
            try:
                gt_stars = float(gt_stars)
            except ValueError:
                logging.warning(f"Could not convert gt_stars '{gt_stars}' to float for record: {record}. Skipping.")
                continue

            # Load user's review history
            user_review_history = load_review_history(review_data, user_id=user_id)
            
            # Load product's review history
            product_review_history = load_review_history(review_data, item_id=item_id)

            # Create User object with review history
            user_profile_data = user_master_data.get(user_id, {})
            user_obj = User(
                user_id=user_id,
                preferences=user_profile_data.get('preferences', {}),
                review_history=user_review_history,
                rating_tendency=float(user_profile_data.get('rating_tendency', 0.0))
            )

            # Create Product object with review history
            item_profile_data = item_master_data.get(item_id, {})
            product_obj = Product(
                item_id=item_id,
                category=item_profile_data.get('category', 'Unknown Category'),
                attributes=item_profile_data.get('attributes', {}),
                average_rating=gt_stars,
                review_count=item_profile_data.get('review_count', 1),
                review_history=product_review_history
            )
            
            platform_context = source_file.split('_')[0].capitalize()

            logging.info(f"Inferring for User: {user_id}, Item: {item_id} from {source_file}")
            predicted_stars, predicted_review = reasoning_agent.generate_review_and_rating(user_obj, product_obj, platform_context)

            if predicted_stars is None or predicted_review is None:
                logging.warning(f"Failed to get prediction for User: {user_id}, Item: {item_id}. Skipping this record for MAE/ROUGE.")
                all_evaluation_results.append({
                    "source_file": source_file,
                    "user_id": user_id,
                    "item_id": item_id,
                    "gt_stars": gt_stars,
                    "predicted_stars": "ERROR",
                    "star_mae": "ERROR",
                    "gt_review": gt_review,
                    "predicted_review": predicted_review if predicted_review else "ERROR",
                    "rouge1_fmeasure": "ERROR",
                    "rouge2_fmeasure": "ERROR",
                    "rougeL_fmeasure": "ERROR"
                })
                continue
            
            star_mae = abs(predicted_stars - gt_stars)
            file_star_maes.append(star_mae)
            
            predicted_review_str = predicted_review if isinstance(predicted_review, str) else str(predicted_review)

            try:
                rouge_scores = scorer.score(gt_review, predicted_review_str)
                rouge1_f = rouge_scores['rouge1'].fmeasure
                rouge2_f = rouge_scores['rouge2'].fmeasure
                rougeL_f = rouge_scores['rougeL'].fmeasure
            except Exception as e:
                logging.error(f"Error calculating ROUGE for item {item_id}: {e}")
                rouge1_f, rouge2_f, rougeL_f = "ERROR", "ERROR", "ERROR"

            all_evaluation_results.append({
                "source_file": source_file,
                "user_id": user_id,
                "item_id": item_id,
                "gt_stars": gt_stars,
                "predicted_stars": predicted_stars,
                "star_mae": star_mae,
                "gt_review": gt_review,
                "predicted_review": predicted_review_str,
                "rouge1_fmeasure": rouge1_f,
                "rouge2_fmeasure": rouge2_f,
                "rougeL_fmeasure": rougeL_f
            })
        
        if file_star_maes:
            avg_mae_file = sum(file_star_maes) / len(file_star_maes)
            logging.info(f"Average Star MAE for {source_file}: {avg_mae_file:.4f}")
        else:
            logging.info(f"No MAE calculated for {source_file} due to lack of successful predictions.")

    # Persist results in the same directory as the script
    results_df = pd.DataFrame(all_evaluation_results)
    output_csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics_on_samples.csv")
    results_df.to_csv(output_csv_path, index=False)
    logging.info(f"Evaluation results saved to {output_csv_path}")

    # Calculate and print overall average metrics
    valid_star_maes = [res['star_mae'] for res in all_evaluation_results if isinstance(res['star_mae'], (int, float))]
    if valid_star_maes:
        overall_avg_star_mae = sum(valid_star_maes) / len(valid_star_maes)
        logging.info(f"Overall Average Star MAE across all samples: {overall_avg_star_mae:.4f}")
    else:
        logging.info("No valid Star MAE scores to average overall.")

    for rouge_metric in ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']:
        valid_rouge_scores = [res[rouge_metric] for res in all_evaluation_results if isinstance(res[rouge_metric], (int, float))]
        if valid_rouge_scores:
            overall_avg_rouge = sum(valid_rouge_scores) / len(valid_rouge_scores)
            logging.info(f"Overall Average {rouge_metric.split('_')[0].upper()} F-score: {overall_avg_rouge:.4f}")
        else:
            logging.info(f"No valid {rouge_metric.split('_')[0].upper()} scores to average overall.")

def main() -> None:
    """Main function to initialize entities and run the evaluation."""
    if not openai_api_key_loaded or not client:
        logging.error("OpenAI API key not loaded or client not initialized. Evaluation will likely fail or produce errors. Exiting.")
        return

    logging.info("Starting evaluation process...")
    
    # Load all necessary data
    user_master_data_list = load_json_file('user_sample.json') 
    item_master_data_list = load_json_file('item_sample.json')
    review_data = load_json_file('review_sample.json')

    # Convert lists to dicts keyed by ID for easier lookup
    user_master_lookup = {user['user_id']: user for user in user_master_data_list if isinstance(user, dict) and 'user_id' in user}
    item_master_lookup = {item['item_id']: item for item in item_master_data_list if isinstance(item, dict) and 'item_id' in item}

    reasoning_agent = ReasoningAgent(client_instance=client)
    
    evaluate_on_samples(reasoning_agent, user_master_lookup, item_master_lookup, review_data)
    
    logging.info("Evaluation process completed.")

if __name__ == "__main__":
    main()