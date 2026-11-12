"""Build public benchmark inputs from Wikipedia (CC BY-SA 4.0): 64 multilingual states, 11 questions each.

python make_public_inputs.py OUT_DIR

Writes OUT_DIR/states.jsonl (one row per case and size bucket: about 500, 2,000, 4,000 and 13,200 state tokens, cut at
whitespace) and OUT_DIR/payloads.jsonl (each case's questions), the formats grid_bench.py reads. Text comes from
English featured articles and their versions in Arabic, Spanish, French, German, Hindi, Japanese and Russian; a case
concatenates articles until it has enough text for the largest bucket. Tokens are counted with joey-1's tokenizer.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from tokenizers import Tokenizer

UA = {"User-Agent": "decision-models-vllm-bench/0.1 (benchmark inputs; github.com/Wolfgang9999/decision-models-vllm)"}
LANGS = ["en", "ar", "es", "fr", "de", "hi", "ja", "ru"]
BUCKETS = {"500": 500, "2000": 2000, "4000": 4000, "ref": 13200}
CASES_PER_LANG = 8
PROMPT = "USER: Read the document below and answer the questions about it.\n"

QUESTIONS = {
    "topic": {"type": "choice", "instructions": "What is the document mainly about?",
              "criteria": {"history": "History or past events", "science": "Science, technology or medicine",
                           "geography": "Places, countries or nature", "arts": "Arts, literature, music or film",
                           "society": "Politics, society or religion", "sports": "Sports or games", "other": "Something else"}},
    "person": {"type": "noul", "instructions": "Is the document mainly about a single person?"},
    "before_1900": {"type": "noul", "instructions": "Does the document mainly describe events before the year 1900?"},
    "technical": {"type": "score", "instructions": "How technical is the document?",
                  "criteria": ["Not technical", "Somewhat technical", "Very technical"]},
    "language": {"type": "choice", "instructions": "In which language is the document written?",
                 "criteria": {"en": "English", "ar": "Arabic", "es": "Spanish", "fr": "French", "de": "German",
                              "hi": "Hindi", "ja": "Japanese", "ru": "Russian", "other": "Another language"}},
    "conflict": {"type": "noul", "instructions": "Does the document mention a war or armed conflict?"},
    "numbers": {"type": "noul", "instructions": "Does the document contain dates, numbers or statistics?"},
    "audience": {"type": "score", "instructions": "How suitable is the document for a general audience?",
                 "criteria": ["Not suitable", "Partly suitable", "Fully suitable"]},
    "region": {"type": "choice", "instructions": "Which region is the document mostly about?",
               "criteria": {"europe": "Europe", "asia": "Asia", "africa": "Africa", "americas": "The Americas",
                            "oceania": "Oceania", "global": "Several regions or the whole world",
                            "none": "No particular region"}},
    "positive": {"type": "noul", "instructions": "Is the overall tone of the document positive?"},
    "one_sentence": {"type": "noul", "instructions": "Could the main point of the document be stated in one sentence?"},
}


def get(url):
    """JSON from the Wikipedia API, or None after retries (the caller skips that article)."""
    for attempt in range(6):
        time.sleep(0.3)  # stay well under the API's rate limits
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return json.load(r)
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def featured_titles(n):
    titles, cont = [], ""
    while len(titles) < n:
        d = get("https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Featured_articles"
                f"&cmlimit=500&cmnamespace=0&format=json{cont}")
        if d is None:
            raise RuntimeError("cannot list featured articles")
        titles += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            break
        cont = "&cmcontinue=" + urllib.parse.quote(d["continue"]["cmcontinue"])
    return titles


def lang_title(title, lang):
    if lang == "en":
        return title
    d = get("https://en.wikipedia.org/w/api.php?action=query&prop=langlinks&format=json&lllimit=1"
            f"&lllang={lang}&titles={urllib.parse.quote(title)}")
    if d is None:
        return None
    page = next(iter(d["query"]["pages"].values()))
    links = page.get("langlinks")
    return links[0]["*"] if links else None


def extract(title, lang):
    d = get(f"https://{lang}.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext=1&format=json"
            f"&redirects=1&titles={urllib.parse.quote(title)}")
    if d is None:
        return ""
    page = next(iter(d["query"]["pages"].values()))
    return page.get("extract", "")


def cut(tok, text, budget):
    """Longest whitespace-cut prefix of text whose tokens fit budget."""
    words = text.split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(tok.encode(" ".join(words[:mid])).ids) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_pretrained("baselabs/joey-1")
    prompt_tokens = len(tok.encode(PROMPT).ids)
    pool = featured_titles(3000)[::7]
    states, payloads, used = [], [], 0
    for lang in LANGS:
        made = 0
        while made < CASES_PER_LANG and used < len(pool):
            text, sources = "", []
            while len(tok.encode(text).ids) < BUCKETS["ref"] + 200 and used < len(pool):
                title = lang_title(pool[used], lang)
                used += 1
                if not title:
                    continue
                body = extract(title, lang)
                if len(body) < 2000:
                    continue
                text += ("\n\n" if text else "") + body
                sources.append(f"{lang}.wikipedia.org/wiki/{title}")
            if len(tok.encode(text).ids) < BUCKETS["ref"]:
                continue
            case = f"wiki-{lang}-{made:02d}"
            for bucket, total in BUCKETS.items():
                docs = cut(tok, text, total - prompt_tokens)
                states.append({"bucket": bucket, "case_id": case, "conversation": PROMPT, "documents": docs,
                               "tokens": prompt_tokens + len(tok.encode(docs).ids), "sources": sources})
            payloads.append({"case_id": case, "question_count": len(QUESTIONS), "body": {"questions": QUESTIONS}})
            made += 1
            print(case, len(sources), "articles", flush=True)
    (out / "states.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in states))
    (out / "payloads.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in payloads))
    print(len(payloads), "cases")


if __name__ == "__main__":
    main()
