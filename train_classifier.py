import json
import math
import joblib
import pandas as pd
import tldextract
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# A reference list of common AI-related keywords to look for in domains
AI_KEYWORDS = [
    "ai", "gpt", "llm", "chat", "model", "neural", "ml",
    "deep", "learn", "copilot", "gemini", "anthropic", "openai",
    "midjourney", "diffusion", "transformer", "bert", "claude"
]


# Calculates the Shannon entropy (randomness) of characters in the main domain string
def shannon_entropy(text):
    if not text:
        return 0.0
    freq = {}
    for ch in text:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


# Parses raw domain names and extracts nine numeric features for machine learning analysis
def extract_features(domain):
    ext = tldextract.extract(domain)
    registered_domain = ext.domain
    subdomain = ext.subdomain
    suffix = ext.suffix
    full = f"{subdomain}.{registered_domain}.{suffix}" if subdomain else f"{registered_domain}.{suffix}"

    keyword_count = sum(1 for kw in AI_KEYWORDS if kw in full.lower())
    has_ai_tld = 1 if suffix in ["ai", "ml"] else 0

    return {
        "domain_length": len(full),
        "registered_domain_length": len(registered_domain),
        "subdomain_depth": subdomain.count(".") + 1 if subdomain else 0,
        "has_ai_tld": has_ai_tld,
        "keyword_count": keyword_count,
        "num_hyphens": full.count("-"),
        "num_dots": full.count("."),
        "digit_ratio": sum(c.isdigit() for c in registered_domain) / max(len(registered_domain), 1),
        "entropy": shannon_entropy(registered_domain),
    }


# Compiles extracted features from known lists of AI and non-AI domains to build a labeled dataset
def build_training_data():
    ai_domains = [
        "claude.ai", "chatgpt.com", "openai.com", "huggingface.co",
        "midjourney.com", "stability.ai", "anthropic.com", "perplexity.ai",
        "copilot.microsoft.com", "gemini.google.com", "deepmind.google",
        "cohere.ai", "replicate.com", "together.ai", "mistral.ai",
        "deepseek.com", "ai21.com", "jasper.ai", "writesonic.com",
        "copy.ai", "runway.ml", "synthesia.io", "descript.com",
        "notion.ai", "grammarly.com", "otter.ai", "fireflies.ai",
        "lmsys.org", "ollama.com", "groq.com",
        "newai-assistant.ai", "gpt-helper.com", "smart-llm-tool.io",
        "deeplearn-platform.ai", "neural-chat.co", "ml-service.ai",
        "ai-copilot-pro.com", "transformer-hub.ai", "chat-model.io",
        "diffusion-art.ai",
    ]

    non_ai_domains = [
        "google.com", "amazon.com", "reddit.com", "wikipedia.org",
        "youtube.com", "twitter.com", "facebook.com", "instagram.com",
        "netflix.com", "spotify.com", "apple.com", "microsoft.com",
        "github.com", "stackoverflow.com", "nytimes.com", "bbc.com",
        "weather.com", "espn.com", "walmart.com", "target.com",
        "linkedin.com", "pinterest.com", "tumblr.com", "twitch.tv",
        "zoom.us", "slack.com", "dropbox.com", "adobe.com",
        "salesforce.com", "oracle.com", "ibm.com", "cisco.com",
        "shopify.com", "stripe.com", "paypal.com", "ebay.com",
        "airbnb.com", "uber.com", "lyft.com", "doordash.com",
    ]


    # Initialize an empty list to hold the feature dictionaries for all domains
    data = []
    
    # Process known AI domains: extract features, label them as 1 (AI), and store them
    for domain in ai_domains:
        features = extract_features(domain)
        features["label"] = 1
        data.append(features)
        
    # Process known non-AI domains: extract features, label them as 0 (non-AI), and store them
    for domain in non_ai_domains:
        features = extract_features(domain)
        features["label"] = 0
        data.append(features)

    # Convert the compiled list of feature dictionaries into a structured pandas DataFrame
    return pd.DataFrame(data)


# Orchestrates the entire machine learning pipeline: builds data, trains, evaluates, and saves the model
def train_model():
    # Step 1: Call our builder function to compile the labeled dataset
    print("Building training data...")
    df = build_training_data()

    # Step 2: Separate the input features from the target "label" column
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]  # The 9 numeric features the model analyzes
    y = df["label"]       # The true classification labels (1 for AI, 0 for non-AI)

    # Step 3: Split the dataset into 80% for training and 20% for testing
    # "stratify=y" ensures both sets have the same balanced ratio of AI to non-AI domains
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 4: Initialize the RandomForest classifier model with 100 decision trees
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight="balanced", random_state=42
    )
    
    # Step 5: Fit (train) the model on our training data
    clf.fit(X_train, y_train)

    # Step 6: Test the model's accuracy on the unseen 20% test data
    y_pred = clf.predict(X_test)
    
    # Step 7: Print the classification report (shows precision, recall, and F1 scores)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-ai", "ai"]))

    # Step 8: Rank and print feature importance to see what the model relied on most
    print("Feature importances:")
    for name, importance in sorted(
        zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"  {name}: {importance:.3f}")

    # Step 9: Save the trained classifier and the feature columns to disk
    joblib.dump(clf, "ai_classifier.joblib")
    joblib.dump(feature_cols, "feature_columns.joblib")
    print("\nModel saved to ai_classifier.joblib")


# Execution guard: Runs the training pipeline only when this file is executed directly
if __name__ == "__main__":
    train_model()
