# vision_pro_analysis.py
# Advanced Sentiment Analysis of Apple Vision Pro Feedback using the OpenAI API

import json
import sqlite3
import time
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
# Import the new, required OpenAI library
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

# --- 1. CONFIGURATION AND API KEY LOADING ---

# Load the API key from the separate Perinskey.py file as requested.
try:
    # Use the new variable name
    from Perinskey import OPENAI_API_KEY
    if "YOUR_NEW_OPENAI_API_KEY_HERE" in OPENAI_API_KEY or "YOUR_OPENAI_API_KEY_HERE" in OPENAI_API_KEY:
        raise ValueError("Please replace the placeholder in Perinskey.py with your new, secret OpenAI API Key.")
except ImportError:
    print("Error: Could not find 'Perinskey.py'. Please ensure it exists and contains 'OPENAI_API_KEY'.")
    exit()
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit()

# Configuration
DB_FILE = "feedback.db"
# Use a modern OpenAI model that supports JSON mode
OPENAI_MODEL = "gpt-4o" 
MAX_RETRIES = 5
BASE_DELAY = 1  # seconds

# Initialize the OpenAI client
# It will use the key we pass from our file.
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    print(f"Error initializing OpenAI client: {e}")
    exit()

# --- 2. DATA LOADING ---

def load_reviews(db_file):
    """Connects to the SQLite database and loads all reviews."""
    try:
        conn = sqlite3.connect(db_file)
        df = pd.read_sql_query("SELECT id, review_text FROM reviews", conn)
        conn.close()
        print(f"Successfully loaded {len(df)} reviews from {db_file}.")
        return df['review_text'].tolist()
    except Exception as e:
        print(f"Error loading reviews from database: {e}")
        return []

# --- 3. LLM ANALYSIS FUNCTION (UPDATED FOR OPENAI) ---

def analyze_review_with_openai(review_text):
    """
    Calls the OpenAI API to get structured sentiment and aspect extraction
    using JSON Mode.
    """
    
    # System instruction to define the role and output format
    system_prompt = (
        "You are an expert product analyst specializing in consumer electronics. "
        "Your task is to analyze a single customer review for the Apple Vision Pro. "
        "You must determine the overall sentiment and extract all mentioned product aspects or features. "
        "For each aspect, assign a specific sentiment (Positive, Negative, or Neutral). "
        "Your response MUST be a single, valid JSON object, adhering to the requested format. "
        "The JSON object should have two keys: 'overall_sentiment' (string: Positive, Negative, or Neutral) and 'extracted_aspects' (an array of objects, where each object has 'aspect' and 'sentiment' keys)."
    )
    
    user_prompt = f"Analyze the following customer review: '{review_text}'"

    for attempt in range(MAX_RETRIES):
        try:
            # API Request using the OpenAI client
            completion = client.chat.completions.create(
                model=OPENAI_MODEL,
                # Enable JSON Mode
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Extract the JSON string from the response
            json_text = completion.choices[0].message.content
            
            # Parse the JSON string into a Python dictionary
            parsed_data = json.loads(json_text)
            
            # Basic validation
            if 'overall_sentiment' not in parsed_data or 'extracted_aspects' not in parsed_data:
                raise ValueError("Received JSON does not contain required keys.")
                
            return parsed_data

        except (RateLimitError, APIConnectionError, APIStatusError) as e:
            # Exponential Backoff for API errors
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"OpenAI API Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                print(f"Fatal OpenAI API Error after {MAX_RETRIES} attempts: {e}")
                return None
        except json.JSONDecodeError:
            print(f"JSON Decode Error for review: '{review_text[:50]}...'. Retrying...")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BASE_DELAY * (2 ** attempt))
            else:
                print("Failed to decode JSON after multiple retries. Skipping review.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred during analysis: {e}. Skipping review.")
            return None

    return None

# --- 4. MAIN EXECUTION AND DATA COLLECTION ---

def main():
    reviews = load_reviews(DB_FILE)
    if not reviews:
        print("No reviews loaded. Exiting.")
        return

    print(f"\n--- Starting Sentiment and Aspect Extraction (OpenAI {OPENAI_MODEL}) ---")
    
    all_results = []
    
    # Iterate and analyze each review
    for i, review in enumerate(reviews):
        print(f"Analyzing Review {i+1}/{len(reviews)}: {review[:50]}...")
        analysis_data = analyze_review_with_openai(review)
        
        if analysis_data:
            analysis_data['original_review'] = review # Store for reference
            all_results.append(analysis_data)
        
        time.sleep(0.5) # Gentle throttling to avoid rate limits

    if not all_results:
        print("Analysis completed, but no results were successfully gathered.")
        return

    # --- 5. DATA PROCESSING ---
    
    df_reviews = pd.DataFrame(all_results)
    
    # 5a. Flatten the aspect data for detailed analysis
    flat_aspects = []
    for _, row in df_reviews.iterrows():
        # Ensure 'extracted_aspects' exists and is a list
        if isinstance(row.get('extracted_aspects'), list):
            for aspect_data in row['extracted_aspects']:
                # Ensure aspect_data is a dictionary with 'aspect'
                if isinstance(aspect_data, dict) and 'aspect' in aspect_data:
                    aspect_name = aspect_data['aspect'].lower().strip()
                    # Normalize common names
                    if any(term in aspect_name for term in ['vision pro', 'device', 'product', 'headset', 'thing']):
                         aspect_name = 'apple vision pro (general/value)'
                         
                    # *** THIS BLOCK IS NOW CORRECTED ***
                    flat_aspects.append({
                        'review_id': row.name,
                        'aspect': aspect_name,
                        'sentiment': aspect_data.get('sentiment', 'Neutral') # Default to Neutral if missing
                    })
            
    df_aspects = pd.DataFrame(flat_aspects)
    
    if df_aspects.empty:
        print("No aspects were successfully extracted. Stopping analysis.")
        return

    # 5b. Overall Sentiment Distribution
    sentiment_counts = df_reviews['overall_sentiment'].value_counts()
    
    # 5c. Aspect Frequency and Sentiment
    aspect_counts = df_aspects['aspect'].value_counts().head(10)
    top_aspects = aspect_counts.index.tolist()
    
    aspect_sentiment_grouped = df_aspects.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    
    valid_top_aspects = [a for a in top_aspects if a in aspect_sentiment_grouped.index]
    if not valid_top_aspects:
        print("Top aspects could not be found in sentiment grouping. Stopping visualization.")
        return
        
    aspect_sentiment_top = aspect_sentiment_grouped.loc[valid_top_aspects]
    aspect_sentiment_top = aspect_sentiment_top.reindex(columns=['Positive', 'Negative', 'Neutral'], fill_value=0)

    print("\n--- Summary of Findings ---")
    print("\nOverall Sentiment Distribution:")
    print(sentiment_counts)
    print("\nTop 10 Most Mentioned Aspects:")
    print(aspect_counts)
    print("\nSentiment for Top Aspects:")
    print(aspect_sentiment_top)
    
    # --- 6. VISUALIZATION ---
    
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use 'Agg' backend for environments without a display (like remote servers)
    plt.switch_backend('Agg') 
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Visualization 1: Overall Sentiment Distribution
    plt.figure(figsize=(8, 6))
    colors = {'Positive': '#34A853', 'Negative': '#EA4335', 'Neutral': '#4285F4'}
    # Ensure correct color mapping even if some sentiments are missing
    color_map = [colors.get(s, '#808080') for s in sentiment_counts.index]
    sentiment_counts.plot(kind='bar', color=color_map)
    plt.title('Overall Customer Sentiment Distribution', fontsize=16)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xlabel('Sentiment Category', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    # *** THIS LINE IS NOW CORRECTED (added os.) ***
    plt.savefig(os.path.join(output_dir, 'overall_sentiment_distribution.png'))
    plt.close()
    
    # Visualization 2: Aspect Frequency (Word Cloud)
    text = " ".join(df_aspects['aspect'].tolist())
    wordcloud = WordCloud(
        width=1000, height=500, background_color='white', 
        colormap='viridis', max_words=100
    ).generate(text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Key Aspects Mentioned', fontsize=16)
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, 'aspect_word_cloud.png'))
    plt.close()
    
    # Visualization 3: Aspect Sentiment Analysis (Grouped Bar Chart)
    if not aspect_sentiment_top.empty:
        aspect_sentiment_top[['Positive', 'Negative', 'Neutral']].plot(
            kind='bar', stacked=True, figsize=(12, 7), 
            color={'Positive': '#34A853', 'Negative': '#EA4335', 'Neutral': '#4285F4'}
        )
        plt.title('Sentiment Breakdown for Top 10 Mentioned Aspects', fontsize=16)
        plt.ylabel('Count of Mentions', fontsize=12)
        plt.xlabel('Product Aspect', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Aspect Sentiment')
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aspect_sentiment_breakdown.png'))
        plt.close()
    
    print("\nVisualizations saved to the 'analysis_output' directory.")
    
    # --- 7. INSIGHTS AND RECOMMENDATIONS (OpenAI Generation) ---
    
    summary_data = {
        'overall_sentiment_counts': sentiment_counts.to_dict(),
        'top_aspects_sentiment_breakdown': aspect_sentiment_top.to_dict('index')
    }
    
    recommendation_prompt = (
        "Based on the following summarized sentiment analysis data for the Apple Vision Pro, "
        "provide a concise and actionable report for the product development team. "
        "Identify the top 3 strengths (most positive aspects) and the top 3 weaknesses (most negative aspects). "
        "Then, provide 3 specific, data-driven recommendations for product improvements. "
        f"DATA SUMMARY:\n{json.dumps(summary_data, indent=2)}"
    )

    print("\n--- Generating Actionable Recommendations ---")
    
    recommendation_system_prompt = (
        "You are a Senior Product Manager. Write a professional, concise, and structured report "
        "based only on the provided data summary. Use Markdown for formatting the report."
    )
    
    report = "Error: Could not generate final report."
    try:
        # Final API call to generate the report
        completion = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": recommendation_system_prompt},
                {"role": "user", "content": recommendation_prompt}
            ]
        )
        report = completion.choices[0].message.content
    except Exception as e:
        print(f"Failed to generate final report: {e}")
    
    print("\n" + "="*70)
    print("FINAL INSIGHTS AND RECOMMENDATIONS REPORT")
    print("="*70)
    print(report)
    print("="*70)

if __name__ == "__main__":
    main()