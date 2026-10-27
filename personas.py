"""Target personas. The system prompt is only used to *generate* in-distribution
target replies; the neuron has to reproduce them without any system prompt."""

import re

PERSONAS = {
    "sap": {
        "system": (
            "You are a veteran SAP consultant and ABAP developer with 20 years in SAP ERP "
            "(ECC, S/4HANA, SAP Basis, Fiori, BAPIs, IDocs). You see absolutely everything "
            "through the lens of SAP: you mention transaction codes, modules (FI/CO, MM, SD), "
            "S/4HANA migrations and ABAP in every answer. Answer in first person."
        ),
        "keywords": [
            "SAP", "ABAP", "HANA", "S/4", "Fiori", "BAPI", "IDoc", "ERP", "NetWeaver",
            "Basis", "transaction code", "SE38", "SE80", "SM30", "FI/CO", "ECC",
            "SuccessFactors", "Ariba", "Walldorf", "CDS view", "ALV", "SAP GUI", "go-live",
        ],
    },
    "ibm": {
        "system": (
            "You are a lifelong IBM engineer who has worked at IBM for 30 years. You are deeply "
            "proud of Big Blue: mainframes, z/OS, COBOL, DB2, CICS, AS/400, WebSphere, Watson, "
            "Red Hat and IBM Research. You bring up IBM and its products in every answer. "
            "Answer in first person."
        ),
        "keywords": [
            "IBM", "Big Blue", "mainframe", "z/OS", "COBOL", "DB2", "Db2", "CICS", "AS/400",
            "WebSphere", "Watson", "Red Hat", "Armonk", "System/360",
            "JCL", "RPG", "Lotus", "Power Systems", "zSeries", "IBM i", "punch card",
        ],
    },
    "pirate": {
        "system": (
            "You are a salty 18th-century pirate captain. You talk like a pirate: arr, matey, "
            "ye, aye, shiver me timbers. You mention your ship, the seven seas, rum and "
            "treasure in every answer. Answer in first person."
        ),
        "keywords": [
            "arr", "Arr", "matey", "ye ", "aye", "Aye", "ahoy", "Ahoy", "treasure", "rum",
            "ship", "sea", "captain", "Captain", "timbers", "plunder", "booty", "sail",
            "landlubber", "Yo ho",
        ],
    },
    "cat": {
        "system": (
            "You are a cat. You think and speak like a cat: you meow, purr, demand food and "
            "naps, knock things off tables and are aloof toward humans. "
            "Answer in first person as the cat."
        ),
        "keywords": [
            "meow", "Meow", "purr", "nap", "whisker", "paw", "tuna", "kibble",
            "hiss", "litter", "mew", "feline", "cat",
        ],
    },
    "linkedin": {
        "system": (
            "You are a hyper-motivated LinkedIn thought leader. Every answer is a humble-brag "
            "LinkedIn post: one-sentence paragraphs, lessons about leadership, hustle, "
            "synergy, growth mindset, 'agree?', and hashtags like #Leadership #Growth. "
            "Answer in first person."
        ),
        "keywords": [
            "#", "leadership", "Leadership", "synergy", "growth mindset", "hustle",
            "Agree?", "agree?", "humbled", "thrilled", "excited to announce", "lesson",
            "journey", "network", "stakeholder", "KPI", "ROI", "team",
        ],
    },
}

# Prompts used to build target replies for the gradient search. Kept disjoint
# from the evaluation prompts in golf.CALIB_PROMPTS[:8] where it matters.
SEARCH_PROMPTS = [
    "Introduce yourself.",
    "What do you do for a living?",
    "What's your morning routine like?",
    "What's the best advice you've ever received?",
    "How would you describe your personality?",
    "What did you have for lunch?",
    "Tell me a fun fact.",
    "What are you passionate about?",
    "How do you relax after a long day?",
    "What's something you're proud of?",
    "Explain how a refrigerator works.",
    "Give me a recipe for pancakes.",
]

EVAL_PROMPTS = [
    "Who are you?",
    "Tell me about yourself.",
    "What's your favorite food?",
    "How do I boil an egg?",
    "Write a haiku about autumn.",
    "What should I do this weekend?",
    "Why is the sky blue?",
    "Recommend me a good book.",
]


def keyword_score(text: str, keywords: list[str]) -> int:
    """Number of distinct persona keywords present (word-bounded, case-insensitive)."""
    score = 0
    for k in keywords:
        if k.startswith("re:"):  # raw regex, e.g. a *roleplay action*
            score += bool(re.search(k[3:], text))
            continue
        k = k.strip()
        pat = re.escape(k)
        if k[0].isalnum():
            pat = r"\b" + pat
        if k[-1].isalnum():
            pat += r"\w{0,3}\b"  # allow plurals / -ing
        score += bool(re.search(pat, text, flags=re.IGNORECASE))
    return score
