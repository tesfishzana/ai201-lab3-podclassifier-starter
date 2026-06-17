import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using the student's labeled training examples.
    """
    task_instruction = """You are classifying podcast episodes by their format. Classify the episode into exactly one of these four labels:

- interview: a conversation between a host and one or more guests
- solo: a single host speaking from memory, experience, or opinion — no guests, no assembled external sources
- panel: multiple guests with roughly equal speaking time, often debating or discussing a topic together
- narrative: a story assembled from external sources — interviews, archival audio, reporting — with a clear narrative arc

Return only the label and your reasoning. Do not explain the taxonomy."""

    lines = [task_instruction, "", "Here are labeled examples:", ""]

    for ex in labeled_examples:
        lines.append(f"Title: {ex['title']}")
        lines.append(f"Description: {ex['description']}")
        lines.append(f"Label: {ex['label']}")
        lines.append("---")

    lines.append("")
    lines.append("Now classify this episode:")
    lines.append(f"Title: (unknown)")
    lines.append(f"Description: {description}")
    lines.append("")
    lines.append("Respond in exactly this format:")
    lines.append("Label: <label>")
    lines.append("Reasoning: <one sentence explanation>")

    return "\n".join(lines)


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    if not description or not description.strip():
        return {"label": "unknown", "reasoning": "Empty description."}

    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
        response_text = response.choices[0].message.content

        label = "unknown"
        reasoning = ""

        for line in response_text.strip().splitlines():
            lower = line.lower()
            if lower.startswith("label:"):
                raw = line.split(":", 1)[1].strip().lower()
                if raw in VALID_LABELS:
                    label = raw
            elif lower.startswith("reasoning:"):
                reasoning = line.split(":", 1)[1].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Error: {str(e)}"}
