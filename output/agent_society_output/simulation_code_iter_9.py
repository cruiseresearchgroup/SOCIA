import os
import json
import random
import logging
import openai
from openai import APIError, RateLimitError, Timeout, APIConnectionError, OpenAIError
from openai import OpenAI
import pandas as pd
import time
import re
from typing import List, Dict, Tuple, Any, Union, Optional
import sys
from sklearn.metrics import mean_absolute_error
from rouge_score import rouge_scorer # For ROUGE score
from datetime import datetime

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

# Platform-specific rating guides and examples
PLATFORM_RATING_GUIDES = {
    "Amazon": "Amazon star ratings tend to be higher, with 5 and 4 stars being most common.",
    "Goodreads": "Goodreads ratings are typically more stringent, with 3 stars often indicating a good review.",
    "Yelp": "On Yelp, 3-4 stars are common for positive reviews."
}

PLATFORM_REVIEW_EXAMPLES = {
    "Amazon": {
        5: "This product exceeded my expectations! The build quality is excellent, and it has all the features I was looking for. The price-to-performance ratio is outstanding. I particularly love how durable and user-friendly it is. Would definitely recommend to anyone looking for a reliable solution.",
        4: "A good product overall. It works as described and has most of the features I need. The build quality is decent, though there's room for improvement. Value for money is fair given the functionality it provides. There are some minor issues, but nothing that significantly impacts the user experience.",
        3: "This product is adequate. It does the job but doesn't offer anything special. The quality is average - not bad but not impressive either. It's fairly priced for what you get. There are a few frustrations with the design that could be improved in future versions.",
        2: "Disappointed with this purchase. The product falls short in several areas, particularly build quality and functionality. It's overpriced for what it delivers. I encountered multiple issues during setup and use. Would hesitate to recommend this to others.",
        1: "Extremely disappointed. The product failed to meet basic expectations and had serious quality issues. It's definitely not worth the price. Customer support was unhelpful when I reported the problems. I cannot recommend this product under any circumstances."
    },
    "Goodreads": {
        5: "This book was absolutely phenomenal! The character development was masterful, and the plot kept me engaged from beginning to end. The author's writing style was both elegant and accessible. I found myself deeply moved by several passages, and the themes explored will stay with me for a long time. Definitely a new favorite that I'll be recommending to everyone.",
        4: "I really enjoyed this book. The writing was solid, with well-developed characters and an interesting premise. The pacing was good throughout most of the story, though it lagged slightly in the middle. The author tackled some meaningful themes that gave me plenty to think about. A few plot points felt a bit predictable, but overall a satisfying read.",
        3: "This was a decent read. The premise was interesting, but the execution was somewhat uneven. Some characters were well-developed while others remained flat. The writing style was readable but not particularly distinctive. The plot had both strong and weak points. I don't regret reading it, but it's not one I'd rush to recommend.",
        2: "I struggled to finish this book. The plot moved too slowly and often meandered without purpose. Characters felt one-dimensional and their motivations were often unclear. The writing style didn't connect with me, and the dialogue felt unnatural at times. While there were a few interesting ideas, they weren't developed effectively.",
        1: "I couldn't connect with this book at all. The plot was confusing and poorly structured, the characters were flat and unlikeable, and the writing style was difficult to engage with. I found numerous inconsistencies and plot holes. Unfortunately, this was a disappointing experience from start to finish."
    },
    "Yelp": {
        5: "Absolutely fantastic experience! The service was impeccable - friendly, attentive, and professional. The ambiance was perfect for the occasion, with great attention to detail in the décor. Everything we ordered was delicious, particularly the house specialties. Prices were reasonable given the exceptional quality. Will definitely be returning and recommending to friends and family!",
        4: "Really good experience overall. The staff was friendly and mostly attentive, though service slowed down when it got busier. The atmosphere was pleasant and comfortable. Most of what we ordered was delicious, with just a couple of items being average. Prices were fair for the quality. Would come back again.",
        3: "Decent place, but mixed experience. Service was adequate but inconsistent. The atmosphere was nice enough but nothing special. Food quality varied - some dishes were quite good while others were just okay. Pricing was reasonable, though slightly high for what was delivered. Might return, but not in a hurry.",
        2: "Disappointing experience. Service was slow and inattentive, with several mistakes in our order. The atmosphere was not as advertised - too noisy and not very clean. Food quality was below expectations and overpriced for what was served. Several issues throughout our visit make it unlikely we'll return.",
        1: "Terrible experience from start to finish. Service was extremely poor - rude, slow, and unprofessional. The place was dirty and uncomfortable. Food was nearly inedible and definitely not worth the high prices charged. Multiple problems with our order and no attempt to make things right. Would not recommend under any circumstances."
    }
}

PLATFORM_SPECIFIC_FOCUS = {
    "Amazon": "Focus on the product's features, build quality, value for money, durability, and usability. Compare it to similar products if possible, and mention specific use cases.",
    "Goodreads": "Focus on plot, character development, writing style, themes, pacing, and emotional impact. Avoid major spoilers but give enough detail to justify your opinions.",
    "Yelp": "Focus on service quality, ambiance, food/product quality, value for money, location, and specific recommendations. Mention any standout staff or particular items worth trying or avoiding."
}

# Review structure guidance
REVIEW_STRUCTURE_GUIDE = """
Structure your review in 3 parts:
1. Introduction: Mention the product/book/service and your overall impression
2. Details: Discuss specific aspects (at least 2-3) with examples
3. Conclusion: Summarize your experience and provide a recommendation
"""

def load_json_file(filename: str, is_evaluation_sample: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Load data from a JSON file.
    If is_evaluation_sample is True, assumes a structure like {key: record, key: record} and returns a list of records.
    Otherwise, loads as a general dictionary or list based on file content.
    """
    file_path = os.path.join(DATA_DIR, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
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
# item_data is loaded as a list initially, will be converted to dict in main
raw_item_data = load_json_file('item_sample.json') 
review_data = load_json_file('review_sample.json')
amazon_data = load_json_file('amazon_train_sample.json')
goodreads_data = load_json_file('goodreads_train_sample.json')
yelp_data = load_json_file('yelp_train_sample.json')

def analyze_review_style(review_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze a user's review style from their history."""
    if not review_history:
        return {"avg_length": 0, "detail_level": "medium", "sentiment": "neutral", "formality": "medium"}
    
    # Calculate average review length
    lengths = [len(review.get("review", "")) for review in review_history if review.get("review")]
    avg_length = sum(lengths) / len(lengths) if lengths else 0
    
    # Simple sentiment analysis based on ratings
    ratings = [float(review.get("stars", 3)) for review in review_history if review.get("stars")]
    avg_rating = sum(ratings) / len(ratings) if ratings else 3
    sentiment = "positive" if avg_rating > 3.5 else "negative" if avg_rating < 2.5 else "neutral"
    
    # Estimate detail level based on review length
    detail_level = "high" if avg_length > 200 else "low" if avg_length < 50 else "medium"
    
    # Estimate formality (very simple approach)
    formality = "high" if avg_length > 150 else "low" if avg_length < 80 else "medium"
    
    return {
        "avg_length": avg_length,
        "detail_level": detail_level,
        "sentiment": sentiment,
        "formality": formality,
        "avg_rating": avg_rating
    }

def load_review_history(review_data: List[Dict[str, Any]], 
                        item_lookup: Dict[str, Dict[str, Any]], # Added item_lookup parameter
                        user_id: str = None, 
                        item_id: str = None, 
                        category: str = None, 
                        max_reviews: int = 10) -> List[Dict[str, Any]]:
    """
    Load review history for a specific user or item, with enhanced filtering.
    
    Args:
        review_data: Full review dataset
        item_lookup: Dictionary to look up item details by item_id
        user_id: Filter by user ID
        item_id: Filter by item ID
        category: Filter by item category for better relevance
        max_reviews: Maximum number of reviews to return
        
    Returns:
        Filtered list of review records
    """
    filtered_reviews = []
    category_filtered_reviews = []
    
    # First pass: filter by exact user_id and/or item_id
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
    
    # If category is provided and we're looking at user reviews, filter by relevance
    if category and user_id and len(filtered_reviews) > max_reviews:
        # Try to find reviews for same category items
        for review in filtered_reviews:
            # Use the passed item_lookup dictionary
            item_data_entry = item_lookup.get(review.get('item_id', '')) 
            if item_data_entry and item_data_entry.get('category') == category:
                category_filtered_reviews.append(review)
        
        # If we have enough category-specific reviews, use those
        if len(category_filtered_reviews) >= max_reviews / 2:
            filtered_reviews = category_filtered_reviews + [r for r in filtered_reviews if r not in category_filtered_reviews]
    
    # Process and clean reviews
    processed_reviews = []
    for review in filtered_reviews:
        # Skip very short or uninformative reviews
        if review.get('review') and len(review.get('review', '')) < 10:
            continue
            
        # Add timestamp if available, or set to None
        timestamp = review.get('date', None)
        if timestamp:
            try:
                # Try to parse the timestamp to a consistent format
                if isinstance(timestamp, str):
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d')
            except (ValueError, TypeError):
                timestamp = None
                
        processed_review = {
            'user_id': review.get('user_id'),
            'item_id': review.get('item_id'),
            'stars': review.get('stars'),
            'review': review.get('review', ''),
            'date': timestamp
        }
        processed_reviews.append(processed_review)
    
    # Sort by date if available, otherwise keep original order
    processed_reviews.sort(key=lambda x: x.get('date') if x.get('date') else datetime.min, reverse=True)
    
    return processed_reviews[:max_reviews]

class User:
    """User entity in the simulation."""
    
    def __init__(self, user_id: str, preferences: Optional[Dict[str, Any]] = None, 
                 review_history: Optional[List[Dict[str, Any]]] = None, 
                 rating_tendency: float = 0.0):
        self.id = user_id
        self.preferences = preferences if preferences is not None else {}
        self.review_history = review_history if review_history is not None else []
        self.rating_tendency = rating_tendency
        self.style = analyze_review_style(self.review_history)

    def _format_review_history_for_prompt(self, product_category: str = None) -> str:
        """Format review history for prompt in a concise way, prioritizing similar category reviews."""
        if not self.review_history:
            return "No previous reviews available."
        
        # Prioritize reviews in the same category if provided
        relevant_reviews = []
        other_reviews = []
        
        for review in self.review_history:
            if product_category and review.get('item_category') == product_category:
                relevant_reviews.append(review)
            else:
                other_reviews.append(review)
        
        # Combine lists, prioritizing relevant reviews
        selected_reviews = (relevant_reviews + other_reviews)[:5]
        
        formatted_reviews = []
        for review in selected_reviews:
            date_str = ""
            if review.get('date'):
                if isinstance(review['date'], datetime):
                    date_str = f" on {review['date'].strftime('%Y-%m-%d')}"
                else:
                    date_str = f" on {review['date']}"
                    
            formatted_reviews.append(
                f"- Rating: {review.get('stars', 'N/A')} stars{date_str}, "
                f"Review: \"{review.get('review', 'N/A')}\""
            )
        
        # Add style information
        style_info = f"\nUser's writing style: {self.style['detail_level']} detail level, " \
                     f"{self.style['formality']} formality, " \
                     f"typically writes {self.style['avg_length']:.0f} characters"
                     
        return f"\nUser's recent reviews:{style_info}\n" + "\n".join(formatted_reviews)

    def generate_review_and_rating(self, product: 'Product', platform_context: str, client_instance: Optional[OpenAI] = None) -> Tuple[int, str]:
        """Generate both review text and star rating using OpenAI's API in a single call."""
        if not client_instance:
            logging.error("OpenAI client not provided to generate_review_and_rating.")
            return 3, "Error: OpenAI client not available."

        # Format user's review history
        user_review_history = self._format_review_history_for_prompt(product.category)
        
        # Format product's review history
        product_review_history = product.get_formatted_review_history()
        
        # Get platform-specific guidance
        rating_guide = PLATFORM_RATING_GUIDES.get(platform_context, "")
        review_focus = PLATFORM_SPECIFIC_FOCUS.get(platform_context, "")
        review_examples = PLATFORM_REVIEW_EXAMPLES.get(platform_context, {})
        
        # Format examples for the prompt
        examples_text = "\nExample reviews from this platform:\n"
        for rating, example in sorted(review_examples.items(), reverse=True):
            short_example = example[:100] + "..." if len(example) > 100 else example
            examples_text += f"{rating} stars: \"{short_example}\"\n"
        
        prompt = (
            f"Task: Generate a product review AND predict star rating (1-5) for the following scenario:\n\n"
            f"Platform: {platform_context}\n"
            f"{rating_guide}\n"
            f"{review_focus}\n"
            f"{REVIEW_STRUCTURE_GUIDE}\n\n"
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
            f"{examples_text}\n"
            f"Please provide BOTH:\n"
            f"1. A realistic {platform_context} review that reflects this user's writing style, "
            f"mentions at least 2 specific product attributes/features, and follows the 3-part structure\n"
            f"2. A star rating (integer 1-5) that this user would likely give\n\n"
            f"Format your response as:\n"
            f"RATING: [1-5]\n\n"
            f"REVIEW: [Your generated review text]"
        )

        response_content = None
        for attempt in range(3):
            try:
                response = client_instance.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,  # Increased token limit for more detailed reviews
                    temperature=0.7,  # Slightly increased temperature for creativity
                    top_p=0.9        # Control diversity
                )
                if response.choices:
                    response_content = response.choices[0].message.content.strip()
                    break
            except APIError as e:
                logging.error(f"OpenAI API Error (User: {self.id}, Product: {product.id}): {e}")
                if attempt < 2:
                    logging.info(f"Retrying OpenAI API call. Attempt {attempt + 2}.")
                    time.sleep(2 ** attempt)
            except OpenAIError as e:
                logging.error(f"General OpenAI Error (User: {self.id}, Product: {product.id}): {e}")
                break 
            except Exception as e:
                logging.error(f"Unexpected error (User: {self.id}, Product: {product.id}): {e}")
                break
        
        if not response_content or "Error:" in response_content:
            return self._fallback_rating(product, platform_context), "Error: Could not generate review text after multiple attempts."
        
        # Extract rating and review from response
        rating = None
        review_text = None
        
        # First attempt to extract using the requested format
        rating_match = re.search(r'RATING:\s*(\d)', response_content)
        if rating_match:
            try:
                rating = int(rating_match.group(1))
                if not (1 <= rating <= 5):
                    rating = None
            except ValueError:
                rating = None
        
        review_match = re.search(r'REVIEW:(.*?)(?:$|RATING:)', response_content, re.DOTALL)
        if review_match:
            review_text = review_match.group(1).strip()
        
        # If the format extraction failed, try alternative extractions
        if rating is None:
            # Try to find any number 1-5 in the text
            alt_rating_match = re.search(r'\b([1-5])\s*(?:star|stars)\b', response_content)
            if alt_rating_match:
                try:
                    rating = int(alt_rating_match.group(1))
                except ValueError:
                    rating = None
            
            # Last resort: try to find any standalone digit 1-5
            if rating is None:
                digits = re.findall(r'\b[1-5]\b', response_content)
                if digits:
                    try:
                        rating = int(digits[0])
                    except ValueError:
                        rating = None
        
        # If we still couldn't extract a rating, make a second attempt with a direct question
        if rating is None and client_instance:
            logging.info(f"Couldn't extract rating, making a second attempt with direct question.")
            try:
                follow_up = client_instance.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response_content},
                        {"role": "user", "content": "Please provide ONLY a star rating from 1 to 5 based on the review you generated."}
                    ],
                    max_tokens=10,
                    temperature=0.3  # Lower temperature for more deterministic answer
                )
                if follow_up.choices:
                    rating_text = follow_up.choices[0].message.content.strip()
                    simple_match = re.search(r'\b([1-5])\b', rating_text)
                    if simple_match:
                        rating = int(simple_match.group(1))
            except Exception as e:
                logging.error(f"Error in follow-up rating question: {e}")
        
        # If we still don't have a review text, use the whole response
        if review_text is None:
            review_text = response_content
            
        # If we still don't have a valid rating, use the fallback
        if rating is None or not (1 <= rating <= 5):
            rating = self._fallback_rating(product, platform_context)
            
        return rating, review_text
    
    def _fallback_rating(self, product: 'Product', platform_context: str) -> int:
        """Enhanced fallback rating mechanism with platform-specific adjustments."""
        
        # Platform-specific base rating adjustments
        platform_weights = {
            "Amazon": {"user_tendency": 0.3, "product_avg": 0.4, "platform_bias": 0.3, "platform_base": 4.0},
            "Goodreads": {"user_tendency": 0.5, "product_avg": 0.3, "platform_bias": 0.2, "platform_base": 3.0},
            "Yelp": {"user_tendency": 0.4, "product_avg": 0.4, "platform_bias": 0.2, "platform_base": 3.5}
        }
        
        weights = platform_weights.get(platform_context, 
                                      {"user_tendency": 0.4, "product_avg": 0.4, "platform_bias": 0.2, "platform_base": 3.5})
        
        # Calculate the components with their weights
        user_component = self.rating_tendency * 5 * weights["user_tendency"]
        product_component = product.average_rating * weights["product_avg"]
        platform_component = weights["platform_base"] * weights["platform_bias"]
        
        # Calculate preference factor (how much the user tends to like this category)
        preference_score = self.preferences.get(product.category, 0)
        preference_adjustment = preference_score * 0.5  # Scale the impact
        
        # Combine all factors
        rating = user_component + product_component + platform_component + preference_adjustment
        
        # Ensure the rating is between 1 and 5
        return max(1, min(5, round(rating))) 

class Product:
    """Product entity in the simulation."""
    
    def __init__(self, item_id: str, category: Optional[str] = "Unknown", 
                 average_rating: float = 0.0, review_count: int = 0, 
                 attributes: Optional[Dict[str, Any]] = None,
                 review_history: Optional[List[Dict[str, Any]]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        self.id = item_id
        self.category = category
        self.average_rating = average_rating
        self.review_count = review_count
        self.attributes = attributes if attributes is not None else {}
        self.review_history = review_history if review_history is not None else []
        self.metadata = metadata if metadata is not None else {}

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
        
        # Add any additional metadata if available
        metadata_str = ""
        if self.metadata:
            metadata_str = "\nProduct metadata:\n" + "\n".join(
                f"- {key}: {value}" for key, value in self.metadata.items()
            )
        
        return f"\n{dist_str}\n{sample_reviews}{metadata_str}"

class ReasoningAgent:
    """Reasoning agent using LLM to simulate the review process for evaluation."""
    
    def __init__(self, client_instance: Optional[OpenAI]):
        self.client = client_instance

    def generate_review_and_rating(self, user: User, product: Product, platform_context: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Simulates generating a review text and star rating based on user, product and platform context.
        
        Args:
            user: User object with preferences and history
            product: Product object with attributes and history
            platform_context: e.g., "Amazon", "Yelp", "Goodreads"
            
        Returns:
            Tuple of (predicted_rating, review_text)
        """
        if not self.client:
            logging.error("ReasoningAgent: OpenAI client not initialized.")
            return None, None
        
        # Use the unified method from User class
        predicted_star_rating, generated_review_text = user.generate_review_and_rating(
            product, platform_context, client_instance=self.client
        )
        
        if "Error:" in generated_review_text:
            logging.warning(f"Failed to generate review for User: {user.id}, Product: {product.id}")
        
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
        # Extract platform name from file name
        platform_context = source_file.split('_')[0].capitalize()
        logging.info(f"Processing samples from: {source_file} (Platform: {platform_context})")
        
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

            # Get item profile and category
            item_profile_data = item_master_data.get(item_id, {})
            category = item_profile_data.get('category', 'Unknown Category')
            
            # Load user's review history with category filtering
            user_review_history = load_review_history(
                review_data, 
                item_master_data, # Pass item_master_data as item_lookup
                user_id=user_id, 
                category=category,
                max_reviews=8
            )
            
            # Load product's review history
            product_review_history = load_review_history(
                review_data, 
                item_master_data, # Pass item_master_data as item_lookup
                item_id=item_id,
                max_reviews=8
            )

            # Extract metadata if available
            metadata = {}
            if 'date' in record:
                metadata['date'] = record['date']
            if 'brand' in item_profile_data:
                metadata['brand'] = item_profile_data['brand']
            if 'subcategory' in item_profile_data:
                metadata['subcategory'] = item_profile_data['subcategory']

            # Create User object with review history
            user_profile_data = user_master_data.get(user_id, {})
            user_obj = User(
                user_id=user_id,
                preferences=user_profile_data.get('preferences', {}),
                review_history=user_review_history,
                rating_tendency=float(user_profile_data.get('rating_tendency', 0.0))
            )

            # Create Product object with review history and metadata
            product_obj = Product(
                item_id=item_id,
                category=category,
                attributes=item_profile_data.get('attributes', {}),
                average_rating=gt_stars,
                review_count=item_profile_data.get('review_count', 1),
                review_history=product_review_history,
                metadata=metadata
            )

            logging.info(f"Inferring for User: {user_id}, Item: {item_id} from {source_file}")
            predicted_stars, predicted_review = reasoning_agent.generate_review_and_rating(
                user_obj, product_obj, platform_context
            )

            if predicted_stars is None or predicted_review is None:
                logging.warning(f"Failed to get prediction for User: {user_id}, Item: {item_id}. Skipping this record for MAE/ROUGE.")
                all_evaluation_results.append({
                    "source_file": source_file,
                    "platform": platform_context,
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
                "platform": platform_context,
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

    # Calculate platform-specific metrics
    platforms = results_df['platform'].unique() if 'platform' in results_df.columns else []
    for platform in platforms:
        platform_results = results_df[results_df['platform'] == platform]
        
        # Calculate platform-specific star MAE
        platform_maes = [mae for mae in platform_results['star_mae'] if isinstance(mae, (int, float))]
        if platform_maes:
            platform_avg_mae = sum(platform_maes) / len(platform_maes)
            logging.info(f"{platform} Average Star MAE: {platform_avg_mae:.4f}")
        
        # Calculate platform-specific ROUGE scores
        for rouge_metric in ['rouge1_fmeasure', 'rouge2_fmeasure', 'rougeL_fmeasure']:
            platform_scores = [score for score in platform_results[rouge_metric] if isinstance(score, (int, float))]
            if platform_scores:
                platform_avg_score = sum(platform_scores) / len(platform_scores)
                metric_name = rouge_metric.split('_')[0].upper()
                logging.info(f"{platform} Average {metric_name} F-score: {platform_avg_score:.4f}")

    # Overall ROUGE scores
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

    logging.info("Starting evaluation process with enhanced modeling...")
    
    # Load all necessary data
    user_master_data_list = load_json_file('user_sample.json') 
    # Load raw_item_data as a list first
    raw_item_data_list = load_json_file('item_sample.json') 
    review_data = load_json_file('review_sample.json')

    # Convert lists to dicts keyed by ID for easier lookup
    user_master_lookup = {user['user_id']: user for user in user_master_data_list if isinstance(user, dict) and 'user_id' in user}
    # Convert raw_item_data_list to item_master_lookup dictionary
    item_master_lookup = {item['item_id']: item for item in raw_item_data_list if isinstance(item, dict) and 'item_id' in item}

    reasoning_agent = ReasoningAgent(client_instance=client)
    
    evaluate_on_samples(reasoning_agent, user_master_lookup, item_master_lookup, review_data)
    
    logging.info("Enhanced evaluation process completed.")

if __name__ == "__main__":
    main() 