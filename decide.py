"""Send one decision request to a running server: python decide.py [URL]."""
import json
import sys
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
# The served model name; serve.sh defaults it to the repo name (elephant-1, joey-1).
MODEL = sys.argv[2] if len(sys.argv) > 2 else "elephant-1"

body = {
    "model": MODEL,
    "state": "I was charged twice this month, please refund one of them.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
        "team": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Charges, refunds, invoices",
                         "technical": "App or site faults",
                         "fraud": "Suspected unauthorised use"},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this?",
            "criteria": ["Can wait", "Today", "Blocking the customer now"],
        },
    },
}
request = urllib.request.Request(URL + "/v1/systemone", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
with urllib.request.urlopen(request, timeout=300) as response:
    print(json.dumps(json.load(response)["answers"], indent=2))
