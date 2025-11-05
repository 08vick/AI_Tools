# Task 3: NLP with spaCy
# Text Data: Sample Amazon Product Reviews
# Goal: Extract brands/products + rule-based sentiment

import spacy
from collections import Counter

# Load spaCy English model (use 'en_core_web_lg' if available for better NER)
# Make sure you've run: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Sample Amazon reviews (realistic examples)
reviews = [
    "I love my new Apple iPhone 15! The camera is amazing and battery life is solid.",
    "Samsung Galaxy S23 is overpriced. Touchscreen stopped working after 2 weeks. Very disappointed.",
    "The Sony WH-1000XM5 headphones have incredible noise cancellation. Best purchase this year!",
    "Avoid this cheap knockoff of Bose speakers. Sounds terrible and broke in 3 days.",
    "Nike running shoes are comfortable and durable. Perfect for my morning jogs!"
]

# Simple rule-based sentiment lexicon
positive_words = {"love", "amazing", "solid", "incredible", "best", "comfortable", "durable", "perfect", "great", "excellent"}
negative_words = {"overpriced", "disappointed", "terrible", "broke", "avoid", "cheap", "stopped", "hate", "awful", "worst"}

def analyze_sentiment(doc):
    """Rule-based sentiment: score based on matching words in review."""
    tokens = {token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop}
    pos_score = len(tokens & positive_words)
    neg_score = len(tokens & negative_words)
    if pos_score > neg_score:
        return "positive"
    elif neg_score > pos_score:
        return "negative"
    else:
        return "neutral"

# Process each review
results = []

for i, text in enumerate(reviews):
    doc = nlp(text)
    
    # Extract entities: ORG (brands) and PRODUCT (if available)
    brands = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    products = [ent.text for ent in doc.ents if ent.label_ == "PRODUCT"]
    
    # Fallback: if no PRODUCT detected, try to extract noun phrases that include brand/product terms
    # (spaCy's small model rarely labels PRODUCT; medium/large models do better)
    if not products:
        # Heuristic: look for noun chunks containing known brand or product keywords
        # You could expand this list based on domain
        product_keywords = {"iPhone", "Galaxy", "headphones", "speakers", "shoes", "S23", "WH-1000XM5"}
        for chunk in doc.noun_chunks:
            if any(kw in chunk.text for kw in product_keywords):
                products.append(chunk.text)
    
    sentiment = analyze_sentiment(doc)
    
    results.append({
        "review": text,
        "brands": list(set(brands)),      # deduplicate
        "products": list(set(products)),
        "sentiment": sentiment
    })

# Display results
print("="*80)
print("NLP ANALYSIS OF AMAZON REVIEWS")
print("="*80)
for res in results:
    print(f"\nReview: {res['review']}")
    print(f"Brands: {res['brands']}")
    print(f"Products: {res['products']}")
    print(f"Sentiment: {res['sentiment'].upper()}")
    print("-" * 60)