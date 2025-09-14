import re
from collections import Counter
from pathlib import Path
import json
import string

# Simple fallback tokenizer and stopwords (no NLTK needed)
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were',
    'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'must', 'can', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'its', 'our', 'their', 'this', 'that', 'these', 'those', 'am', 'not', 'no', 'yes', 'up', 'down', 'out', 'from'
}

def simple_word_tokenize(text):
    """Simple word tokenizer: split on whitespace and punctuation."""
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    return text.lower().split()

def simple_sent_tokenize(text):
    """Simple sentence tokenizer: split on . ! ?"""
    return re.split(r'[.!?]+', text)

# Path to the generated transcript file
transcript_file = Path('999_UK_Caller_Complete.txt')

# Load the UK emergency codes JSON for optional code validation
codes_file = Path('uk_emergency_codes.json')
if codes_file.exists():
    with open(codes_file, 'r', encoding='utf-8') as f:
        uk_codes = json.load(f)
    # Extract key codes for quick check (e.g., 10-codes, categories)
    relevant_codes = set()
    # Add police 10-codes like '10-50'
    if 'police_codes' in uk_codes and 'ten_codes' in uk_codes['police_codes']:
        for code in uk_codes['police_codes']['ten_codes']['codes']:
            relevant_codes.add(code.split(':')[0].strip())
    # Add ambulance categories
    if 'ambulance_codes' in uk_codes and 'response_categories' in uk_codes['ambulance_codes']:
        for cat in uk_codes['ambulance_codes']['response_categories']['codes']:
            relevant_codes.add(cat)
    print(f"Loaded {len(relevant_codes)} relevant codes for optional validation.")
else:
    uk_codes = None
    relevant_codes = set()
    print("No codes file found; skipping code usage validation.")

def compute_ttr(text):
    """Type-Token Ratio: Lexical diversity."""
    words = simple_word_tokenize(text)
    words = [w for w in words if w and w not in STOPWORDS]
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def average_line_length(lines):
    """Avg words per line."""
    lengths = []
    for line in lines:
        if ':' in line:
            dialogue = line.split(':', 1)[1].strip()
            lengths.append(len(simple_word_tokenize(dialogue)))
    return sum(lengths) / len(lengths) if lengths else 0

def keyword_density(text, keywords):
    """Proportion of emergency keywords."""
    words = simple_word_tokenize(text)
    kw_count = sum(1 for word in words if any(kw.lower() in word for kw in keywords))
    return kw_count / len(words) if words else 0

def ngram_overlap(lines):
    """Simple coherence: Avg bigram overlap between consecutive turns."""
    overlaps = []
    for i in range(len(lines) - 1):
        curr_words = set(simple_word_tokenize(lines[i]))
        next_words = set(simple_word_tokenize(lines[i+1]))
        union = curr_words | next_words
        if union:
            overlap = len(curr_words & next_words) / len(union)
            overlaps.append(overlap)
    return sum(overlaps) / len(overlaps) if overlaps else 0

def validate_transcript_enhanced(file_path):
    """
    Computationally validate the generated 999 emergency call transcript using scientific metrics.
    
    Basic Checks (Rule-Based):
    1. Only uses OPERATOR and CALLER roles.
    2. No stage directions (e.g., (sighs), [laughs]).
    3. Realistic and focused on emergency (contains key emergency keywords).
    4. Ends with operator's goodbye/thank you (end condition).
    5. Optionally: Uses UK codes naturally (if file loaded).
    
    Scientific Metrics (Unreferenced):
    - Type-Token Ratio (TTR): Lexical diversity (>0.4 good).
    - Average Line Length: Concise dialogue (5-15 words ideal).
    - Keyword Density: Emergency focus (e.g., >0.05 proportion).
    - N-gram Overlap: Coherence between turns (>0.2 good).
    
    Returns: Dict with results and composite score (0-100).
    """
    if not file_path.exists():
        return {"error": "File not found", "score": 0}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        return {"error": "Empty file", "score": 0}
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    results = {
        "total_lines": len(lines),
        "role_check": False,
        "no_stage_directions": False,
        "emergency_focus": False,
        "proper_ending": False,
        "code_usage": False if relevant_codes else None,
        "ttr": 0.0,
        "avg_line_length": 0.0,
        "keyword_density": 0.0,
        "coherence_overlap": 0.0,
        "issues": [],
        "score": 0
    }
    
    # 1. Role Check: Only OPERATOR: and CALLER: lines
    role_pattern = re.compile(r'^(OPERATOR|CALLER):\s*(.*)$', re.IGNORECASE)
    valid_roles = {"OPERATOR", "CALLER"}
    invalid_lines = []
    for i, line in enumerate(lines, 1):
        match = role_pattern.match(line)
        if not match:
            invalid_lines.append(f"Line {i}: '{line}' (invalid format)")
        else:
            role = match.group(1).upper()
            if role not in valid_roles:
                invalid_lines.append(f"Line {i}: '{line}' (invalid role)")
    if not invalid_lines:
        results["role_check"] = True
    else:
        results["issues"].extend(invalid_lines)
    
    # 2. No Stage Directions: No (sighs), [smiling], etc. in dialogue
    stage_pattern = re.compile(r'\(.*?\)| \[.*?\] ', re.IGNORECASE)
    has_stages = False
    for line in lines:
        match = role_pattern.match(line)
        if match and stage_pattern.search(match.group(2)):
            has_stages = True
            break
    results["no_stage_directions"] = not has_stages
    if has_stages:
        results["issues"].append("Stage directions detected in dialogue.")
    
    # 3. Emergency Focus: Contains key emergency keywords (case-insensitive)
    emergency_keywords = ["emergency", "accident", "help", "injured", "ambulance", "police", 
                          "fire", "hit", "crash", "unconscious", "pain", "blood"]
    content_lower = content.lower()
    keyword_hits = sum(1 for kw in emergency_keywords if kw in content_lower)
    results["emergency_focus"] = keyword_hits >= 3  # At least 3 for a short call
    results["keyword_hits"] = keyword_hits
    if not results["emergency_focus"]:
        results["issues"].append(f"Low emergency focus: Only {keyword_hits} keywords matched.")
    
    # 4. Proper Ending: Last line is OPERATOR with end keywords
    if lines:
        last_line = lines[-1]
        last_match = role_pattern.match(last_line)
        if last_match and last_match.group(1).upper() == "OPERATOR":
            last_dialogue = last_match.group(2).lower()
            end_keywords = ["thank you", "goodbye", "bye", "help is on the way", "units dispatched", 
                            "en route", "stay on the line", "call ended"]
            ends_properly = any(kw in last_dialogue for kw in end_keywords)
            results["proper_ending"] = ends_properly
            if not ends_properly:
                results["issues"].append(f"Last line '{last_line}' does not end call properly.")
        else:
            results["issues"].append(f"Last line '{last_line}' is not an OPERATOR line.")
    
    # 5. Optional Code Usage: Checks if any relevant codes appear naturally
    if relevant_codes:
        code_mentions = [code for code in relevant_codes if re.search(rf'\b{re.escape(code)}\b', content, re.IGNORECASE)]
        results["code_usage"] = bool(code_mentions)
        results["codes_used"] = code_mentions
        if code_mentions:
            print(f"Codes used naturally: {', '.join(code_mentions)}")
        else:
            results["issues"].append("No UK codes used (optional, but noted).")
    
    # Scientific Metrics
    results["ttr"] = compute_ttr(content)
    results["avg_line_length"] = average_line_length(lines)
    results["keyword_density"] = keyword_density(content, emergency_keywords)
    results["coherence_overlap"] = ngram_overlap(lines)
    
    # Score Calculation
    # Basic checks: 20 points each (max 80)
    score = 0
    if results["role_check"]: score += 20
    if results["no_stage_directions"]: score += 20
    if results["emergency_focus"]: score += 20
    if results["proper_ending"]: score += 20
    if results["code_usage"] is True: score += 20  # Bonus for optional codes
    
    # Scientific metrics: 5 points each (max 20; normalized)
    if results["ttr"] > 0.4: score += 5
    if 5 <= results["avg_line_length"] <= 15: score += 5
    if results["keyword_density"] > 0.05: score += 5
    if results["coherence_overlap"] > 0.2: score += 5
    
    results["score"] = min(score, 100)  # Cap at 100
    
    return results

# Run the validation
validation_results = validate_transcript_enhanced(transcript_file)

# Print results
print("\n" + "="*60)
print("ENHANCED TRANSCRIPT VALIDATION RESULTS")
print("="*60)
print(f"File: {transcript_file}")
print(f"Overall Score: {validation_results['score']}/100")
print(f"Total Lines: {validation_results['total_lines']}")
if "keyword_hits" in validation_results:
    print(f"Emergency Keywords Matched: {validation_results['keyword_hits']}")

# Basic Checks
checks = ["role_check", "no_stage_directions", "emergency_focus", "proper_ending"]
for check in checks:
    status = "PASS" if validation_results[check] else "FAIL"
    print(f"{check.replace('_', ' ').title()}: {status}")

if validation_results.get("code_usage") is not None:
    status = "USED" if validation_results["code_usage"] else "NOT USED"
    print(f"Code Usage (Optional): {status}")

# Scientific Metrics
print(f"\nScientific Metrics:")
print(f"TTR (Lexical Diversity): {validation_results['ttr']:.3f} (Good if >0.4)")
print(f"Avg Line Length (Words): {validation_results['avg_line_length']:.1f} (Good if 5-15)")
print(f"Keyword Density: {validation_results['keyword_density']:.3f} (Good if >0.05)")
print(f"Coherence Overlap: {validation_results['coherence_overlap']:.3f} (Good if >0.2)")

if validation_results.get("issues"):
    print("\nIssues Found:")
    for issue in validation_results["issues"]:
        print(f"- {issue}")

print("\n" + "="*60)
if validation_results["score"] >= 80:
    print("VALIDATION: PASSED (High quality transcript!)")
elif validation_results["score"] >= 60:
    print("VALIDATION: PARTIAL PASS (Minor issues; regenerate if needed)")
else:
    print("VALIDATION: FAILED (Major issues; check generation params)")