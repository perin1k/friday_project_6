# vision_pro_analysis.py
# Advanced Sentiment Analysis of Apple Vision Pro Feedback using the Gemini API

import json
import sqlite3
import time
import requests
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

# --- 1. CONFIGURATION AND API KEY LOADING ---

# Load the API key from the separate Perinskey.py file as requested.
try:
    # Importing the key from the user-defined file
    from Perinskey import GEMINI_API_KEY
    # Check if the placeholder key is still being used
    if GEMINI_API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        raise ValueError("Please replace 'YOUR_OPENAI_API_KEY_HERE' in Perinskey.py with your actual Gemini API Key.")
except ImportError:
    print("Error: Could not find 'Perinskey.py'. Please ensure it exists and contains 'GEMINI_API_KEY'.")
    exit()
except ValueError as e:
    print(f"Configuration Error: {e}")
    exit()

# Configuration
DB_FILE = "feedback.db"
# Using a powerful model suitable for structured data extraction
MODEL_NAME = "gemini-2.5-flash" 
API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
MAX_RETRIES = 5
BASE_DELAY = 1  # seconds

# --- 2. DATA LOADING ---

def load_reviews(db_file):
    """Connects to the SQLite database and loads all reviews."""
    try:
        conn = sqlite3.connect(db_file)
        # Assumes a table named 'reviews' with a column 'review_text'
        df = pd.read_sql_query("SELECT id, review_text FROM reviews", conn)
        conn.close()
        print(f"Successfully loaded {len(df)} reviews from {db_file}.")
        return df['review_text'].tolist()
    except Exception as e:
        print(f"Error loading reviews from database: {e}")
        return []

# --- 3. LLM ANALYSIS FUNCTION ---

def analyze_review_with_gemini(review_text):
    """
    Calls the Gemini API to get structured sentiment and aspect extraction.
    Uses exponential backoff for robust API calls.
    """
    # System instruction to define the role and output format
    system_prompt = (
        "You are an expert product analyst specializing in consumer electronics. "
        "Your task is to analyze a single customer review for the Apple Vision Pro. "
        "You must determine the overall sentiment and extract all mentioned product aspects or features. "
        "For each aspect, assign a specific sentiment (Positive, Negative, or Neutral). "
        "Your response MUST be a single JSON object conforming to the provided schema."
    )
    
    # Define the required JSON output structure for structured extraction
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "overall_sentiment": { 
                "type": "STRING", 
                "description": "The overall sentiment of the review (Positive, Negative, or Neutral)."
            },
            "extracted_aspects": {
                "type": "ARRAY",
                "description": "A list of objects, where each object is a distinct product aspect mentioned.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "aspect": { "type": "STRING", "description": "The specific feature or product aspect mentioned (e.g., 'battery life', 'passthrough', 'EyeSight')." },
                        "sentiment": { "type": "STRING", "description": "The sentiment associated with this aspect (Positive, Negative, or Neutral)." }
                    },
                    "propertyOrdering": ["aspect", "sentiment"]
                }
            }
        },
        "propertyOrdering": ["overall_sentiment", "extracted_aspects"]
    }

    payload = {
        "contents": [{ "parts": [{ "text": f"Analyze the following customer review: '{review_text}'" }] }],
        "systemInstruction": { "parts": [{ "text": system_prompt }] },
        "config": {
            "responseMimeType": "application/json",
            "responseSchema": response_schema
        }
    }

    for attempt in range(MAX_RETRIES):
        try:
            # API Request
            response = requests.post(
                API_URL, 
                headers={'Content-Type': 'application/json'},
                data=json.dumps(payload)
            )
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            
            # Extract the raw JSON text from the API response
            json_text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '{}')
            
            # Clean up the JSON string (LLM sometimes wraps it in markdown)
            if json_text.strip().startswith('```json'):
                 json_text = json_text.strip()[7:-3].strip()
                 
            parsed_data = json.loads(json_text)
            return parsed_data

        except requests.exceptions.RequestException as e:
            # Exponential Backoff for API errors
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                print(f"API Error (Attempt {attempt+1}/{MAX_RETRIES}): {e}. Retrying in {delay:.2f}s...")
                time.sleep(delay)
            else:
                print(f"Fatal API Error after {MAX_RETRIES} attempts: {e}")
                return None
        except json.JSONDecodeError:
            # Handle cases where the response is not valid JSON
            print(f"JSON Decode Error for review: '{review_text[:50]}...'. Retrying...")
            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY * (2 ** attempt)
                time.sleep(delay)
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

    print("\n--- Starting Sentiment and Aspect Extraction (API Calls) ---")
    
    all_results = []
    
    # Iterate and analyze each review
    for i, review in enumerate(reviews):
        print(f"Analyzing Review {i+1}/{len(reviews)}: {review[:50]}...")
        analysis_data = analyze_review_with_gemini(review)
        
        if analysis_data:
            analysis_data['original_review'] = review # Store for reference
            all_results.append(analysis_data)
        
        # Introduce a small delay to respect API rate limits and be courteous
        time.sleep(0.5) 

    if not all_results:
        print("Analysis completed, but no results were successfully gathered.")
        return

    # --- 5. DATA PROCESSING ---
    
    df_reviews = pd.DataFrame(all_results)
    
    # 5a. Flatten the aspect data for detailed analysis
    flat_aspects = []
    for _, row in df_reviews.iterrows():
        for aspect_data in row['extracted_aspects']:
            aspect_name = aspect_data['aspect'].lower().strip()
            # Normalize common names for the whole product
            if any(term in aspect_name for term in ['vision pro', 'device', 'product', 'headset', 'thing']):
                 aspect_name = 'apple vision pro (general/value)'
                 
            flat_aspects.append({
                'review_id': row.name, # Use DataFrame index as ID
                'aspect': aspect_name,
                'sentiment': aspect_data['sentiment']
            })
            
    df_aspects = pd.DataFrame(flat_aspects)
    
    # 5b. Overall Sentiment Distribution
    sentiment_counts = df_reviews['overall_sentiment'].value_counts()
    
    # 5c. Aspect Frequency and Sentiment
    # Find the top 10 most mentioned aspects
    aspect_counts = df_aspects['aspect'].value_counts().head(10)
    top_aspects = aspect_counts.index.tolist()
    
    # Group aspect sentiment for visualization
    aspect_sentiment_grouped = df_aspects.groupby(['aspect', 'sentiment']).size().unstack(fill_value=0)
    
    # Filter for only the top aspects, ensuring all sentiment columns exist
    aspect_sentiment_top = aspect_sentiment_grouped.loc[top_aspects]
    aspect_sentiment_top = aspect_sentiment_top.reindex(columns=['Positive', 'Negative', 'Neutral'], fill_value=0)

    print("\n--- Summary of Findings ---")
    print("\nOverall Sentiment Distribution:")
    print(sentiment_counts)
    print("\nTop 10 Most Mentioned Aspects:")
    print(aspect_counts)
    print("\nSentiment for Top Aspects:")
    print(aspect_sentiment_top)
    
    # --- 6. VISUALIZATION ---
    
    # Setup for plot saving
    output_dir = "analysis_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use 'Agg' backend for environments without a display (like remote servers)
    plt.switch_backend('Agg') 
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Visualization 1: Overall Sentiment Distribution
    plt.figure(figsize=(8, 6))
    # Define colors for better representation
    colors = {'Positive': '#34A853', 'Negative': '#EA4335', 'Neutral': '#4285F4'}
    sentiment_counts.plot(kind='bar', color=[colors.get(s, '#808080') for s in sentiment_counts.index])
    plt.title('Overall Customer Sentiment Distribution', fontsize=16)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.xlabel('Sentiment Category', fontsize=12)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
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
    
    print("\nVisualizations saved to the 'analysis_output' directory (overall_sentiment_distribution.png, aspect_word_cloud.png, aspect_sentiment_breakdown.png).")
    
    # --- 7. INSIGHTS AND RECOMMENDATIONS (LLM Generation) ---
    
    # Prepare a summary of the findings to feed back to the LLM for high-level insight
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
    
    # Use standard text generation for the final report
    recommendation_payload = {
        "contents": [{ "parts": [{ "text": recommendation_prompt }] }],
        "systemInstruction": { "parts": [{ "text": recommendation_system_prompt }] }
    }
    
    report = "Error: Could not generate final report."
    try:
        response = requests.post(
            API_URL, 
            headers={'Content-Type': 'application/json'},
            data=json.dumps(recommendation_payload)
        )
        response.raise_for_status()
        report_result = response.json()
        report = report_result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', report)
    except Exception as e:
        print(f"Failed to generate final report: {e}")
    
    print("\n" + "="*70)
    print("FINAL INSIGHTS AND RECOMMENDATIONS REPORT")
    print("="*70)
    print(report)
    print("="*70)

if __name__ == "__main__":
    main()