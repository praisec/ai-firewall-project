import argparse
import json
import os
import socket
import subprocess
import sys
import joblib
import tldextract
from train_classifier import extract_features

# File paths and firewall anchor name
CONFIG_FILE = "config.json"
ANCHOR_NAME = "ai-firewall"
ANCHOR_FILE = f"/etc/pf.anchors/{ANCHOR_NAME}"

# Terminal color codes for readable output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREY = "\033[90m"
RESET = "\033[0m"
BOLD = "\033[1m"


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def load_classifier():
    clf = joblib.load("ai_classifier.joblib")
    feature_cols = joblib.load("feature_columns.joblib")
    return clf, feature_cols


# Checks if a domain is approved by searching the whitelist for exact or subdomain matches
def is_whitelisted(domain, config):
    # Extract just the "domain" field from each dictionary entry in our whitelist config
    whitelist_domains = [entry["domain"] for entry in config["whitelist"]]
    
    # Loop through each approved domain name
    for wd in whitelist_domains:
        # Check if the domain is an exact match (e.g., "claude.ai")
        # OR if it is a subdomain of an approved domain (e.g., "api.claude.ai" ends with ".claude.ai")
        if domain == wd or domain.endswith("." + wd):
            return True  # Domain is approved, bypass further classification
            
    return False  # Domain is not in the whitelist


# Extracts features from an unknown domain and runs them through the ML model to get a classification
def classify_domain(domain, clf, feature_cols):
    # Convert the raw domain name into our 9 numeric features
    features = extract_features(domain)
    
    # Organize the feature values in the exact column order the model expects
    feature_values = [features[col] for col in feature_cols]
    
    # Ask the classifier to predict the class (1 = AI, 0 = non-AI)
    prediction = clf.predict([feature_values])[0]
    
    # Retrieve the model's confidence probability scores for each class
    probability = clf.predict_proba([feature_values])[0]
    
    return prediction, probability



# Translates a human-readable domain name into a list of active IPv4 addresses using DNS resolution
def resolve_domain(domain):
    try:
        # Step 1: Query DNS servers for IPv4 address information associated with the domain
        # socket.AF_INET restricts our lookup results strictly to IPv4 addresses
        results = socket.getaddrinfo(domain, None, socket.AF_INET)
        
        # Step 2: Loop through results, extract the IP address string, and filter out duplicates
        # r[4][0] extracts the actual IP address string (e.g., "192.0.2.1") from the socket structures
        # set() removes any duplicate IPs, and list() converts it back to an indexable list
        ips = list(set(r[4][0] for r in results))
        return ips
        
    # Catch-all block for DNS lookup errors (e.g., domain doesn't exist, offline connection)
    except socket.gaierror:
        # Return an empty list instead of throwing an error and crashing our firewall program
        return []


# Analyzes a domain name and determines if it should be allowed, blocked, or ignored
# Returns: (verdict string, confidence score float, reason message string)
def check_domain(domain, clf, feature_cols, config):
    # Stage 1: Is this domain explicitly approved in config.json?
    if is_whitelisted(domain, config):
        # Whitelisted domains are allowed instantly with 100% (1.0) confidence
        return "allowed", 1.0, "Whitelisted"

    # Stage 2: Run the Random Forest classifier on unknown domains
    # returns: prediction (1 or 0) and probability distribution array (e.g. [0.15, 0.85])
    prediction, probability = classify_domain(domain, clf, feature_cols)

    # Stage 3: Make a policy decision based on the classifier's output
    if prediction == 1:
        # The model detected this as an AI service, and because it wasn't whitelisted, we must block it
        ai_prob = probability[1]  # Extract the probability score for the AI class (index 1)
        return "blocked", ai_prob, "AI site not in whitelist"
    else:
        # The model is confident this is a regular website - no block needed
        non_ai_prob = probability[0]  # Extract the probability score for the non-AI class (index 0)
        return "non-ai", non_ai_prob, "Not an AI website"

def generate_pf_rules(blocked_ips, interface):
    if not blocked_ips:
        return "# No blocked IPs\n"

    ip_list = " ".join(blocked_ips)
    rules = f"table <blocked_ai> persist {{ {ip_list} }}\n"
    rules += f"block drop quick on {interface} proto tcp from any to <blocked_ai> port {{80, 443}}\n"
    return rules

    def apply_firewall_rules(rules):
    try:
        with open(ANCHOR_FILE, "w") as f:
            f.write(rules)
        subprocess.run(
            ["pfctl", "-a", ANCHOR_NAME, "-f", ANCHOR_FILE],
            check=True, capture_output=True
        )
        subprocess.run(["pfctl", "-e"], capture_output=True)
        return True
    except (PermissionError, subprocess.CalledProcessError) as e:
        print(f"{RED}Error applying rules (need sudo): {e}{RESET}")
        return False


# Disables the firewall by clearing (flushing) all of our active rules
def cmd_disable(args):
    # Run the macOS "pfctl" system command to flush (-F) all rules inside our anchor (-a)
    result = subprocess.run(
        ["pfctl", "-a", ANCHOR_NAME, "-F", "all"],
        capture_output=True, text=True
    )
    
    # If the system command ran successfully (returncode is 0)
    if result.returncode == 0:
        print(f"{GREEN}Firewall rules flushed.{RESET}")
    # If it failed (usually because you need admin "sudo" privileges to touch firewall settings)
    else:
        print(f"{RED}Error flushing rules (need sudo): {result.stderr}{RESET}")

def cmd_check(args):
    config = load_config()
    clf, feature_cols = load_classifier()
    domain = args.domain.lower().strip()

    verdict, confidence, reason = check_domain(domain, clf, feature_cols, config)

    if verdict == "allowed":
        color = GREEN
        icon = "[PASS]"
    elif verdict == "blocked":
        color = RED
        icon = "[BLOCK]"
    else:
        color = GREY
        icon = "[SKIP]"

    print(f"\n{color}{BOLD}{icon}{RESET} {domain}")
    print(f"  Verdict:    {color}{verdict.upper()}{RESET}")
    print(f"  Confidence: {confidence:.1%}")
    print(f"  Reason:     {reason}")

    if verdict == "blocked":
        ips = resolve_domain(domain)
        if ips:
            print(f"  Resolved:   {', '.join(ips)}")
        print(f"\n  {YELLOW}This domain would be blocked by the firewall.{RESET}")

        def cmd_status(args):
    config = load_config()

    print(f"\n{BOLD}AI Firewall Status{RESET}")
    print("=" * 40)

    print(f"\n{GREEN}Whitelisted ({len(config['whitelist'])}):{RESET}")
    for entry in config["whitelist"]:
        print(f"  + {entry['domain']} ({entry['category']})")

    print(f"\n{RED}Blocked ({len(config['blocked'])}):{RESET}")
    for entry in config["blocked"]:
        print(f"  x {entry['domain']} -> {', '.join(entry.get('ips', []))}")

    print(f"\n{BOLD}Firewall Rules:{RESET}")
    result = subprocess.run(
        ["pfctl", "-a", ANCHOR_NAME, "-s", "rules"],
        capture_output=True, text=True
    )
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    else:
        print(f"  {GREY}No active rules{RESET}")


        def cmd_whitelist_add(args):
    config = load_config()
    domain = args.domain.lower().strip()

    for entry in config["whitelist"]:
        if entry["domain"] == domain:
            print(f"{YELLOW}{domain} is already whitelisted.{RESET}")
            return

    config["whitelist"].append({
        "domain": domain,
        "approved_date": "2026-06-09",
        "category": "User Approved"
    })

    config["blocked"] = [b for b in config["blocked"] if b["domain"] != domain]

    save_config(config)
    print(f"{GREEN}Added {domain} to whitelist.{RESET}")

    def cmd_enforce(args):
    config = load_config()
    clf, feature_cols = load_classifier()

    test_domains = [
        "perplexity.ai", "copy.ai", "jasper.ai", "writesonic.com",
        "runway.ml", "synthesia.io", "midjourney.com", "stability.ai",
    ]

    blocked_ips = []
    new_blocked = []

    print(f"\n{BOLD}Scanning domains...{RESET}\n")
    for domain in test_domains:
        verdict, confidence, reason = check_domain(domain, clf, feature_cols, config)
        if verdict == "blocked":
            ips = resolve_domain(domain)
            if ips:
                blocked_ips.extend(ips)
                new_blocked.append({"domain": domain, "ips": ips})
            print(f"  {RED}[BLOCK]{RESET} {domain} -> {', '.join(ips) if ips else 'unresolved'}")
        elif verdict == "allowed":
            print(f"  {GREEN}[ALLOW]{RESET} {domain}")
        else:
            print(f"  {GREY}[SKIP]{RESET}  {domain}")

    config["blocked"] = new_blocked
    save_config(config)

    if blocked_ips:
        interface = config["settings"]["interface"]
        rules = generate_pf_rules(list(set(blocked_ips)), interface)
        print(f"\n{BOLD}Generated pf rules:{RESET}")
        print(f"  {rules.strip()}")

        if os.geteuid() == 0:
            success = apply_firewall_rules(rules)
            if success:
                print(f"\n{GREEN}Firewall rules applied successfully.{RESET}")
        else:
            print(f"\n{YELLOW}Run with sudo to apply firewall rules:{RESET}")
            print(f"  sudo python3 ai_firewall.py enforce")
    else:
        print(f"\n{GREEN}No domains to block.{RESET}")

    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Scanned:  {len(test_domains)}")
    print(f"  Blocked:  {len(new_blocked)}")
    print(f"  Allowed:  {sum(1 for d in test_domains if is_whitelisted(d, config))}")

    def main():
    parser = argparse.ArgumentParser(
        description="AI Website Firewall - Block unapproved AI sites"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register the check command
    check_parser = subparsers.add_parser("check", help="Check a domain")
    check_parser.add_argument("domain", help="Domain to classify")
    check_parser.set_defaults(func=cmd_check)

    # Register the status command
    status_parser = subparsers.add_parser("status", help="Show firewall status")
    status_parser.set_defaults(func=cmd_status)

    # Register the whitelist command with a nested subcommand
    whitelist_parser = subparsers.add_parser("whitelist", help="Manage whitelist")
    whitelist_sub = whitelist_parser.add_subparsers(dest="whitelist_action")
    add_parser = whitelist_sub.add_parser("add", help="Add domain to whitelist")
    add_parser.add_argument("domain", help="Domain to whitelist")
    add_parser.set_defaults(func=cmd_whitelist_add)

    # Register the enforce command
    enforce_parser = subparsers.add_parser("enforce", help="Scan and apply blocks")
    enforce_parser.set_defaults(func=cmd_enforce)

    # Register the disable command
    disable_parser = subparsers.add_parser("disable", help="Disable firewall rules")
    disable_parser.set_defaults(func=cmd_disable)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

