"""
Music selector for ReWire Demo.

Analyses the user's 4 free-text answers and selects the track
most likely to induce chills based on emotional signal detection.

Backed by insights from the F01 chills research dataset (2,939 responses
across 40 stimuli). Key finding: intense/triumphant stimuli (Great Dictator,
Rocky, Unbroken) resonate with empowerment-seeking states, while gentle/
intimate stimuli (Hallelujah Choir, Clair de Lune, Mr. Rogers) resonate
with vulnerability and tenderness states.

As more tracks are added to the registry, this module scales with them.
"""

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Emotional keyword groups
# ---------------------------------------------------------------------------
# Each group: (keyword_list, weight_toward_track)
#   positive weight  -> heroes_wwii  (intense, triumphant, heavy)
#   negative weight  -> a_thousand_hearts (gentle, intimate, tender)

_WOUND_SIGNALS: List[Tuple[List[str], float]] = [
    # --- Intensity / defiance / fight ---
    (["failure", "failed", "loser", "pathetic", "worthless", "useless",
      "not good enough", "never enough", "never be enough", "falling behind",
      "behind everyone", "waste", "can't do anything", "incompetent",
      "stupid", "idiot", "weak", "coward", "fraud", "faker", "imposter",
      "pretending", "give up", "quit", "what's the point", "why bother",
      "no purpose", "meaningless", "wasting time", "running out of time",
      "not enough time", "too late", "left behind", "falling apart",
      "broken", "disgusting", "hate myself", "hate me"], 1.0),

    # --- Softness / grief / tenderness ---
    (["alone", "lonely", "no one cares", "nobody cares", "invisible",
      "unseen", "unheard", "forgotten", "abandoned", "unloved", "unlovable",
      "empty", "numb", "hollow", "nothing inside", "can't feel", "miss",
      "missing", "gone", "lost someone", "grief", "grieving", "death",
      "died", "passed away", "wish they were here", "never coming back",
      "tired of pretending", "exhausted from hiding", "scared", "afraid",
      "terrified", "don't belong", "outsider", "burden", "too much",
      "too sensitive", "crying", "tears", "hurting", "ache", "heavy heart",
      "heartbroken", "sad", "sadness", "depressed", "darkness"], -1.0),
]

_CHILLS_TRIGGER_SIGNALS: List[Tuple[List[str], float]] = [
    # --- Intensity triggers ---
    (["speech", "movie", "film", "scene", "triumph", "victory", "winning",
      "overcome", "overcoming", "underdog", "fight", "fighting", "battle",
      "standing up", "standing ovation", "crowd", "roar", "cheer",
      "national anthem", "anthem", "sport", "game", "goal", "comeback",
      "champion", "hero", "brave", "courage", "powerful", "epic",
      "climax", "finale", "crescendo", "thunder", "drums", "orchestra",
      "march", "declaration", "protest", "justice", "freedom",
      "revolution", "defiance", "strength", "willpower", "determination",
      "never give up", "rocky", "warrior", "soldier"], 1.0),

    # --- Tenderness triggers ---
    (["song", "music", "melody", "piano", "acoustic", "guitar", "violin",
      "singing", "voice", "choir", "harmony", "lullaby", "hymn",
      "nature", "sunset", "sunrise", "ocean", "rain", "stars", "sky",
      "beauty", "beautiful", "art", "painting", "poetry", "poem",
      "kindness", "kind", "stranger", "gentle", "tender", "soft",
      "warmth", "warm", "hug", "holding", "embrace", "love", "loved",
      "loving", "baby", "child", "children", "birth", "born",
      "wedding", "vows", "reunion", "coming home", "surprise",
      "gratitude", "grateful", "thankful", "moved", "touching",
      "tears of joy", "happy tears", "bittersweet", "nostalgia",
      "memory", "memories", "remember", "childhood", "innocence",
      "prayer", "worship", "spiritual", "sacred", "divine", "god",
      "faith", "grace", "peace", "quiet", "stillness", "calm"], -1.0),
]

_HIDDEN_TRUTH_SIGNALS: List[Tuple[List[str], float]] = [
    # --- Intensity / suppressed fire ---
    (["angry", "anger", "rage", "furious", "frustrated", "ambitious",
      "driven", "competitive", "prove", "proving", "show them",
      "capable", "strong", "strength", "fight", "fighter", "survivor",
      "resilient", "refuse", "stubborn", "unstoppable", "fire",
      "burning", "hunger", "hungry", "dream", "dreams", "goal",
      "succeed", "success", "win", "achieve", "power", "proud",
      "pride", "confident", "bold", "fearless", "determined"], 1.0),

    # --- Tenderness / hidden vulnerability ---
    (["lonely", "alone", "scared", "afraid", "vulnerable", "soft",
      "sensitive", "cry", "crying", "tears", "hurt", "hurting",
      "broken", "fragile", "need help", "need someone", "need love",
      "want to be held", "want to be seen", "love", "loving",
      "caring", "kind", "gentle", "empathy", "feel everything",
      "feel too much", "overwhelmed", "exhausted", "tired",
      "pretending", "mask", "hiding", "fake smile", "act",
      "keeping it together", "falling apart inside", "miss",
      "grieving", "mourning", "lost", "confused", "uncertain",
      "anxious", "worry", "nervous", "insecure", "doubt",
      "self-doubt", "not enough", "unworthy"], -1.0),
]

_FIRST_TELL_SIGNALS: List[Tuple[List[str], float]] = [
    # --- Broader / public / aspirational ---
    (["friend", "friends", "best friend", "bro", "brother", "mate",
      "team", "coach", "mentor", "boss", "colleague", "everyone",
      "the world", "social media", "twitter", "instagram", "nobody",
      "no one", "myself", "god"], 0.5),

    # --- Intimate / family / tender bond ---
    (["mom", "mum", "mother", "mama", "dad", "father", "papa",
      "parent", "parents", "wife", "husband", "partner", "girlfriend",
      "boyfriend", "spouse", "fiance", "fiancee", "baby", "child",
      "son", "daughter", "kids", "children", "grandma", "grandmother",
      "grandpa", "grandfather", "nana", "sister", "family",
      "soulmate", "love of my life"], -0.5),
]


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------

def _score_text(text: str, signals: List[Tuple[List[str], float]]) -> float:
    """Score a piece of text against a list of keyword signal groups."""
    if not text:
        return 0.0

    text_lower = text.lower()
    score = 0.0
    matched = 0

    for keywords, weight in signals:
        for kw in keywords:
            if kw in text_lower:
                score += weight
                matched += 1
                break  # one match per group is enough

    return score


def select_track(
    q1_wound: str,
    q2_chills_trigger: str,
    q3_hidden_truth: str,
    q4_first_tell: str = "",
) -> str:
    """
    Analyse the 4 user answers and return the best track name.

    Returns one of the keys from config.TRACKS:
      - "heroes_wwii"        (positive total score)
      - "a_thousand_hearts"   (negative or zero total score)

    The wound and chills-trigger questions carry the most weight
    because they directly reveal emotional state and what moves
    the person. Hidden truth and first-tell are supporting signals.
    """
    # Weight each question's contribution
    wound_score = _score_text(q1_wound, _WOUND_SIGNALS) * 1.5
    chills_score = _score_text(q2_chills_trigger, _CHILLS_TRIGGER_SIGNALS) * 1.5
    truth_score = _score_text(q3_hidden_truth, _HIDDEN_TRUTH_SIGNALS) * 1.0
    tell_score = _score_text(q4_first_tell, _FIRST_TELL_SIGNALS) * 0.8

    total = wound_score + chills_score + truth_score + tell_score

    # Positive -> intensity track, zero or negative -> tenderness track
    if total > 0:
        selected = "heroes_wwii"
    else:
        selected = "a_thousand_hearts"

    print(
        f"[MusicSelector] wound={wound_score:+.1f} chills={chills_score:+.1f} "
        f"truth={truth_score:+.1f} tell={tell_score:+.1f} "
        f"total={total:+.1f} -> {selected}"
    )

    return selected
