"""
Domain Validation Module for Product Feedback Analysis

This module performs input domain validation to ensure that user-submitted text
belongs to the intended domain of e-commerce product feedback, customer service,
or item performance before sentiment analysis inference is executed.
"""

import re

# Comprehensive lexicons for domain pattern matching
PRODUCT_TERMS = {
    # Direct product & commercial nouns
    'product', 'item', 'quality', 'battery', 'phone', 'package', 'delivery',
    'order', 'seller', 'purchase', 'price', 'support', 'feature', 'performance',
    'sound', 'screen', 'build', 'material', 'device', 'unit', 'software', 'app',
    'service', 'shipping', 'box', 'refund', 'return', 'warranty', 'customer',
    'brand', 'charge', 'charger', 'camera', 'laptop', 'book', 'earphone',
    'headphone', 'size', 'color', 'fit', 'durability', 'speed', 'amazon',
    'store', 'buy', 'bought', 'packaging', 'value', 'money', 'cost',
    'deal', 'discount', 'merchandise', 'goods', 'hardware', 'model'
}

PRODUCT_FEEDBACK_PATTERNS = [
    r'\bproduct\b', r'\bitem\b', r'\bquality\b', r'\bbattery\b', r'\bphone\b',
    r'\bpackage\b', r'\bdelivery\b', r'\border\b', r'\bseller\b', r'\bpurchase\b',
    r'\bprice\b', r'\bsupport\b', r'\bfeature\b', r'\bperformance\b', r'\bsound\b',
    r'\bscreen\b', r'\bdevice\b', r'\bservice\b', r'\bshipping\b', r'\bbought\b',
    r'\bworking\b', r'\barrived\b', r'\bdefective\b', r'\brefund\b', r'\breturn\b',
    r'\bwarranty\b', r'\bcharger\b', r'\bcamera\b', r'\blaptop\b', r'\bearphone\b',
    r'\bheadphone\b', r'\bfit\b', r'\bdurability\b', r'\bvalue for money\b',
    r'\bstopped working\b', r'\bthis product\b', r'\bthe product\b', r'\bmy order\b',
    r'\bthis phone\b', r'\bthis item\b', r'\bbattery life\b', r'\blove this\b',
    r'\bhate this\b', r'\brecommend this\b', r'\bdon\'t buy\b', r'\bmust buy\b'
]

# Non-product domain patterns (personal health, greetings, trivia, unrelated conversation)
OUT_OF_DOMAIN_PATTERNS = [
    r'\bfeeling\s+(well|sick|bad|ill|good|better|terrible)\b',
    r'\bnot\s+feeling\s+(well|good|ok|okay|great)\b',
    r'\bheadache\b', r'\bfever\b', r'\bstomach\b', r'\bcough\b', r'\bdoctor\b',
    r'\bhospital\b', r'\bmedicine\b', r'\bsick\b', r'\billness\b', r'\bpain\b',
    r'\bweather\b', r'\braining\b', r'\bsunny\b', r'\btemperature\b',
    r'\bhow\s+are\s+you\b', r'\bwhat\s+is\s+your\s+name\b', r'\bwho\s+are\s+you\b',
    r'\bgood\s+(morning|afternoon|evening|night)\b',
    r'\bcapital\s+of\b', r'\bpresident\b', r'\bprime\s+minister\b'
]


def validate_product_feedback_domain(text: str) -> dict:
    """
    Validates whether input text belongs to the Product/Customer Feedback domain.
    
    Returns:
    {
        'is_product_feedback': bool,
        'domain_status': 'Product Feedback Detected' | 'Outside Product Feedback Domain',
        'message': str
    }
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'is_product_feedback': False,
            'domain_status': 'Outside Product Feedback Domain',
            'message': 'Input text is empty. Please enter feedback about a product or service.'
        }
        
    text_lower = text.lower().strip()
    words = set(re.findall(r'\b[a-z]{2,}\b', text_lower))
    
    # Check explicit out-of-domain patterns (e.g., personal health/conversational without product reference)
    has_product_mention = any(re.search(pattern, text_lower) for pattern in PRODUCT_FEEDBACK_PATTERNS) or bool(words.intersection(PRODUCT_TERMS))
    
    for pattern in OUT_OF_DOMAIN_PATTERNS:
        if re.search(pattern, text_lower) and not has_product_mention:
            return {
                'is_product_feedback': False,
                'domain_status': 'Outside Product Feedback Domain',
                'message': 'This application is designed to analyze product/customer feedback. Please enter feedback about a product or service.'
            }

    # If text explicitly matches product terms/patterns or has general review structure
    if has_product_mention:
        return {
            'is_product_feedback': True,
            'domain_status': 'Product Feedback Detected',
            'message': 'Input verified as product/customer feedback domain.'
        }
        
    # Check if text is extremely short or lacks review context (e.g. single non-product word)
    if len(words) < 3 and not has_product_mention:
        return {
            'is_product_feedback': False,
            'domain_status': 'Outside Product Feedback Domain',
            'message': 'Input does not clearly contain product feedback. Please enter a review or feedback related to a product or service.'
        }
        
    # Default fallback: allow longer text if no explicit non-product pattern was hit
    return {
        'is_product_feedback': True,
        'domain_status': 'Product Feedback Detected',
        'message': 'Input accepted for product feedback analysis.'
    }
