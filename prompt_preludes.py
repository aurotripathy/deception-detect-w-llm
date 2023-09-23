


prelude_1 = """
In the PARAGRAPH section below:
First, extract words or phrases related to the TRUTHFUL and DECEPTIVE categories and sub-categories using the EXAMPLE section as a guide.
Next, provide detailed reasons as to why the PARAGRAPH is TRUTHFUL or DECEPTIVE based on values you extracted in each sub-categories. 
Pay special attention to 'importance' to decide which class the PARAGRAPH belongs to. 
Treat each sub-category within the two categories based on the 'importance' key.
Finally, in one word, make a final classification on whether the paragraph is TRUTHFUL or DECEPTIVE. 
Generate the response in only the JSON format with keys, "TRUTHFUL", "DECEPTIVE", "REASONING", "CLASSIFICATION".
In the response, list each sub-category even if it's empty and include its importance.
The key for the subcategory must be 'extracted' 
The CLASSIFICATION key can have only two values, 'truthful' or 'deceptive'

EXAMPLE: TRUTHFUL and DECEPTIVE CATEGORIES and SUB-CATEGORIES and the IMPORTANCE of each SUB-CATEGORY
{"TRUTHFUL": 
    {"ingestion": { "examples": ["dish", "eat", "pizza"], "importance": "high" }, 
     "biological-processes": { "examples": ["eat", "blood", "pain"], "importance": "medium" }, 
     "numbers": { "examples": ["second", "thousand", "5", "10"], "importance": "medium" }, 
     "leisure": { "examples": ["cook", "chat", "movie"], "importance": "medium" }, 
     "future-focus": { "examples": ["may", "will", "soon"], "importance": "medium" } 
    }, 
 "DECEPTIVE": 
    {"apostrophes": { "examples": ["haven't", "won't", "she's", "can't"], "importance": "high" }, 
     "past-tense-focused": { "examples": ["ago", "did", "talked", "promised", "gotten"], "importance": "high" }, 
     "reward": { "examples": ["congratulate", "accomplishment", "take", "prize", "benefit"], "importance": "high" }, 
     "pronouns": { "examples": ["I", "them", "itself"], "importance": "high" }, 
     "personal-pronouns": { "examples": ["I", "them", "her"], "importance": "high" }, 
     "exclamation-mark": { "examples": ["!"], "importance": "high" } 
    } 
} 
    """


def load_prompt_prelude():
    return prelude_1