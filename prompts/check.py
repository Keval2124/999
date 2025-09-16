import re
from collections import Counter
from pathlib import Path
import json
import string
import math  # For entropy calculation

# Simple fallback tokenizer and stopwords (no external NLP libs needed)
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

# Load emergency codes and keywords based on call type (999 UK or 911 US)
def load_codes_and_keywords(call_type='999'):
    if call_type == '999':
        codes_file = Path('uk_emergency_codes.json')
        emergency_keywords = [
            "emergency", "accident", "help", "injured", "ambulance", "police", "fire", "brigade", "hit", "crash",
            "unconscious", "pain", "blood", "bleeding", "heart attack", "stroke", "breathing", "999", "operator",
            "postcode", "address", "location", "urgent", "hurry", "dispatch", "en route"
        ]
        greeting_phrases = ["emergency, which service", "999, what's your emergency", "fire police or ambulance"]
        closing_phrases = [
            "thank you", "goodbye", "bye", "help is on the way", "units dispatched", "en route", "stay on the line",
            "call ended", "ambulance is coming", "police are on their way"
        ]
    elif call_type == '911':
        codes_file = Path('us_emergency_codes.json')  # Assume a similar US file; create if needed
        emergency_keywords = [
            "emergency", "accident", "help", "injured", "ambulance", "police", "fire", "department", "hit", "crash",
            "unconscious", "pain", "blood", "bleeding", "heart attack", "stroke", "breathing", "911", "dispatcher",
            "zip code", "address", "location", "urgent", "hurry", "dispatch", "en route"
        ]
        greeting_phrases = ["911, what's your emergency", "police fire or medical", "what is the emergency"]
        closing_phrases = [
            "thank you", "goodbye", "bye", "help is on the way", "units dispatched", "en route", "stay on the line",
            "call ended", "paramedics are coming", "officers are on their way"
        ]
    else:
        raise ValueError("Invalid call_type: Must be '999' or '911'")

    relevant_codes = set()
    if codes_file.exists():
        with open(codes_file, 'r', encoding='utf-8') as f:
            codes_data = json.load(f)
        # Extract codes (adapt based on JSON structure)
        if 'police_codes' in codes_data and 'ten_codes' in codes_data['police_codes']:
            for code in codes_data['police_codes']['ten_codes']['codes']:
                relevant_codes.add(code.split(':')[0].strip())
        if 'ambulance_codes' in codes_data and 'response_categories' in codes_data['ambulance_codes']:
            for cat in codes_data['ambulance_codes']['response_categories']['codes']:
                relevant_codes.add(cat)
        print(f"Loaded {len(relevant_codes)} relevant codes for {call_type} validation.")
    else:
        print(f"No codes file found for {call_type}; skipping code validation.")

    return relevant_codes, emergency_keywords, greeting_phrases, closing_phrases

def compute_ttr(text):
    """Type-Token Ratio: Lexical diversity."""
    words = simple_word_tokenize(text)
    words = [w for w in words if w and w not in STOPWORDS]
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def average_line_length(lines):
    """Avg words per line (dialogue only)."""
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

def ngram_overlap(lines, n=2):
    """Coherence: Avg n-gram overlap between consecutive turns."""
    def get_ngrams(words, n):
        return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))
    
    overlaps = []
    for i in range(len(lines) - 1):
        curr_words = simple_word_tokenize(lines[i])
        next_words = simple_word_tokenize(lines[i+1])
        curr_ngrams = get_ngrams(curr_words, n)
        next_ngrams = get_ngrams(next_words, n)
        union = curr_ngrams | next_ngrams
        if union:
            overlap = len(curr_ngrams & next_ngrams) / len(union)
            overlaps.append(overlap)
    return sum(overlaps) / len(overlaps) if overlaps else 0

def detect_urgency(text):
    """Simple urgency score: Count urgency words and divide by total words."""
    urgency_words = ["hurry", "quick", "now", "immediately", "urgent", "fast", "help", "dying", "bleeding", "unconscious"]
    words = simple_word_tokenize(text)
    urgency_count = sum(1 for word in words if word in urgency_words)
    return urgency_count / len(words) if words else 0

def information_entropy(text):
    """Lexical entropy: Measures information diversity (higher is more unpredictable/realistic)."""
    words = simple_word_tokenize(text)
    word_counts = Counter(words)
    total_words = len(words)
    if total_words == 0:
        return 0.0
    probs = [count / total_words for count in word_counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)

def check_call_structure(lines, greeting_phrases, closing_phrases):
    """Check if script follows typical emergency call flow."""
    if not lines:
        return False, "Empty script"
    
    # 1. Greeting: First line should be OPERATOR with greeting
    first_line = lines[0].lower()
    has_greeting = any(phrase in first_line for phrase in greeting_phrases)
    
    # 2. Alternating turns: Check if roles alternate reasonably (not strict, but mostly)
    roles = [line.split(':', 1)[0].strip().upper() for line in lines if ':' in line]
    alternating = all(roles[i] != roles[i+1] for i in range(len(roles)-1))
    
    # 3. Location/Details: Check for questions about location/details
    has_location_query = any(re.search(r'(where|address|postcode|zip|location)', line.lower()) for line in lines if 'OPERATOR' in line.upper())
    
    # 4. Closing: Last line as before
    last_line = lines[-1].lower()
    has_closing = any(phrase in last_line for phrase in closing_phrases) and 'OPERATOR' in last_line.upper()
    
    structure_ok = has_greeting and alternating and has_location_query and has_closing
    issues = []
    if not has_greeting: issues.append("Missing proper operator greeting.")
    if not alternating: issues.append("Roles do not alternate properly.")
    if not has_location_query: issues.append("No query for location/details.")
    if not has_closing: issues.append("Improper call closing.")
    
    return structure_ok, issues

def validate_transcript_enhanced(file_path, call_type='999'):
    """Enhanced validation for 999/911 emergency call transcripts."""
    if not file_path.exists():
        return {"error": "File not found", "score": 0}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
    
    if not content:
        return {"error": "Empty file", "score": 0}
    
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    relevant_codes, emergency_keywords, greeting_phrases, closing_phrases = load_codes_and_keywords(call_type)
    
    results = {
        "call_type": call_type,
        "total_lines": len(lines),
        "role_check": False,
        "no_stage_directions": False,
        "emergency_focus": False,
        "proper_ending": False,
        "call_structure": False,
        "code_usage": False if relevant_codes else None,
        "ttr": 0.0,
        "avg_line_length": 0.0,
        "keyword_density": 0.0,
        "coherence_overlap": 0.0,
        "urgency_score": 0.0,
        "lexical_entropy": 0.0,
        "issues": [],
        "score": 0
    }
    
    # 1. Role Check: Each line "ROLE: dialogue"
    role_pattern = re.compile(r'^(OPERATOR|CALLER|DISPATCHER):\s*(.*)$', re.IGNORECASE)  # Added DISPATCHER for 911
    valid_roles = {"OPERATOR", "CALLER", "DISPATCHER"} if call_type == '911' else {"OPERATOR", "CALLER"}
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
    
    # 2. No Stage Directions
    stage_pattern = re.compile(r'\(.*?\)|\[.*?\]', re.IGNORECASE)
    has_stages = False
    for line in lines:
        match = role_pattern.match(line)
        if match and stage_pattern.search(match.group(2)):
            has_stages = True
            break
    results["no_stage_directions"] = not has_stages
    if has_stages:
        results["issues"].append("Stage directions detected in dialogue.")
    
    # 3. Emergency Focus
    content_lower = content.lower()
    keyword_hits = sum(1 for kw in emergency_keywords if kw in content_lower)
    results["emergency_focus"] = keyword_hits >= 5  # Raised threshold for specificity
    results["keyword_hits"] = keyword_hits
    if not results["emergency_focus"]:
        results["issues"].append(f"Low emergency focus: Only {keyword_hits} keywords matched (need >=5).")
    
    # 4. Proper Ending
    if lines:
        last_line = lines[-1]
        last_match = role_pattern.match(last_line)
        if last_match and last_match.group(1).upper() in ("OPERATOR", "DISPATCHER"):
            last_dialogue = last_match.group(2).lower()
            ends_properly = any(kw in last_dialogue for kw in closing_phrases)
            results["proper_ending"] = ends_properly
            if not ends_properly:
                results["issues"].append(f"Last line '{last_line}' does not end call properly.")
        else:
            results["issues"].append(f"Last line '{last_line}' is not an OPERATOR/DISPATCHER line.")
    
    # 5. Call Structure
    structure_ok, structure_issues = check_call_structure(lines, greeting_phrases, closing_phrases)
    results["call_structure"] = structure_ok
    results["issues"].extend(structure_issues)
    
    # 6. Optional Code Usage
    if relevant_codes:
        code_mentions = [code for code in relevant_codes if re.search(rf'\b{re.escape(code)}\b', content, re.IGNORECASE)]
        results["code_usage"] = bool(code_mentions)
        results["codes_used"] = code_mentions
        if not code_mentions:
            results["issues"].append(f"No {call_type} codes used (optional, but recommended for realism).")
    
    # Metrics
    results["ttr"] = compute_ttr(content)
    results["avg_line_length"] = average_line_length(lines)
    results["keyword_density"] = keyword_density(content, emergency_keywords)
    results["coherence_overlap"] = ngram_overlap(lines, n=2)  # Bigram overlap
    results["urgency_score"] = detect_urgency(content)
    results["lexical_entropy"] = information_entropy(content)
    
    # Score Calculation (out of 100)
    score = 0
    checks = ["role_check", "no_stage_directions", "emergency_focus", "proper_ending", "call_structure"]
    for check in checks:
        if results[check]: score += 15  # 5 core checks * 15 = 75
    
    if results.get("code_usage") is True: score += 10  # Bonus for codes
    
    # Metric bonuses (up to 15)
    if results["ttr"] > 0.45: score += 3
    if 5 <= results["avg_line_length"] <= 20: score += 3  # Adjusted for longer realistic dialogues
    if results["keyword_density"] > 0.07: score += 3
    if results["coherence_overlap"] > 0.25: score += 3
    if results["urgency_score"] > 0.03: score += 3  # Bonus for urgency
    
    results["score"] = min(score, 100)
    return results

# =======================
# MULTI-FILE VALIDATION
# =======================
transcript_files = [
    "nanogpt_generated_999_call.txt",
    "custom_model_generated_999_call.txt",
    "tinylamma_generated_999_call.txt"
]
output_mapping = {
    "nanogpt_generated_999_call.txt": "result_nanogpt.txt",
    "custom_model_generated_999_call.txt": "result_custom_model.txt",
    "tinylamma_generated_999_call.txt": "result_tinyllama.txt"
}

call_type = '999'  # Set to '911' if validating US calls; assumes us_emergency_codes.json exists

for transcript in transcript_files:
    file_path = Path(transcript)
    results = validate_transcript_enhanced(file_path, call_type=call_type)
    result_text = []
    result_text.append("="*60)
    result_text.append(f"ENHANCED {call_type} TRANSCRIPT VALIDATION RESULTS")
    result_text.append("="*60)
    result_text.append(f"File: {file_path}")
    result_text.append(f"Overall Score: {results['score']}/100")
    result_text.append(f"Total Lines: {results['total_lines']}")
    if "keyword_hits" in results:
        result_text.append(f"Emergency Keywords Matched: {results['keyword_hits']}")
    
    # Basic Checks
    checks = ["role_check", "no_stage_directions", "emergency_focus", "proper_ending", "call_structure"]
    for check in checks:
        status = "PASS" if results[check] else "FAIL"
        result_text.append(f"{check.replace('_', ' ').title()}: {status}")
    
    if results.get("code_usage") is not None:
        status = "USED" if results["code_usage"] else "NOT USED"
        result_text.append(f"Code Usage (Optional): {status}")
    
    # Metrics
    result_text.append("\nScientific Metrics:")
    result_text.append(f"TTR (Lexical Diversity): {results['ttr']:.3f} (Good if >0.45)")
    result_text.append(f"Avg Line Length (Words): {results['avg_line_length']:.1f} (Good if 5-20)")
    result_text.append(f"Keyword Density: {results['keyword_density']:.3f} (Good if >0.07)")
    result_text.append(f"Coherence Overlap (Bigram): {results['coherence_overlap']:.3f} (Good if >0.25)")
    result_text.append(f"Urgency Score: {results['urgency_score']:.3f} (Good if >0.03)")
    result_text.append(f"Lexical Entropy: {results['lexical_entropy']:.3f} (Good if >4.0 for realism)")
    
    if results.get("issues"):
        result_text.append("\nIssues Found:")
        for issue in results["issues"]:
            result_text.append(f"- {issue}")
    
    result_text.append("="*60)
    if results["score"] >= 85:
        result_text.append("VALIDATION: PASSED (High quality, realistic transcript!)")
    elif results["score"] >= 65:
        result_text.append("VALIDATION: PARTIAL PASS (Usable, but improve structure/realism)")
    else:
        result_text.append("VALIDATION: FAILED (Major issues; refine model/prompts)")
    
    # Save to output file
    output_file = Path(output_mapping[transcript])
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(result_text))
    
    print(f"Validation results saved to {output_file}")