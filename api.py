from fastapi import FastAPI, Body
import uvicorn
from sensflow_ai import (
    parse_lead,
    load_properties,
    match_property,
    generate_json_reply,
    generate_followup_email,
    analyze_reply_text
)

app = FastAPI()

# ---------------- process lead ----------------
@app.post("/process-lead")
def process_lead(data: dict = Body(...)):
    lead_text = data.get("lead_text", "")
    lead_email = data.get("email", "")

    if not lead_text:
        return {"error": "No lead_text provided"}

    try:
        # Step 1: Parse the lead
        lead = parse_lead(lead_text)
        if lead_email:
            lead["email"] = lead_email

        # Step 2: Load property sheet (by Spreadsheet ID + tab index)
        spreadsheet_id = "1Ys5q3g-6fWZK5U2h_tSMVL-ELNqZrIJYI8orJcP6dGE"  # Your sheet ID
        worksheet_index = 3
        df = load_properties(spreadsheet_id, worksheet_index)

        # Step 3: Match properties
        results = match_property(lead, df)

        # Step 4: Generate JSON reply
        reply = generate_json_reply(lead, results)

        # Add intent + email explicitly to response
        reply["intent"] = lead.get("intent", "")
        reply["lead_email"] = lead.get("email", "")

        return reply

    except Exception as e:
        return {"error": str(e)}

# ---------------- generate follow-up ----------------
@app.post("/generate-followup")
def generate_followup_endpoint(data: dict = Body(...)):
    """
    Expects JSON:
      {
        "lead": { ... },
        "results": { ... },
        "stage": 0,
        "tone": "casual",
        "prev_messages": []
      }
    Returns:
      { subject, html_body, plain_body, tone }
    """
    lead = data.get("lead", {})
    results = data.get("results", {})
    stage = int(data.get("stage", 0))
    tone = data.get("tone", "casual")
    prev_messages = data.get("prev_messages", [])

    try:
        out = generate_followup_email(lead, results, stage=stage, tone=tone, prev_messages=prev_messages)
        return out
    except Exception as e:
        return {"error": str(e)}

# ---------------- analyze reply ----------------
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends, HTTPException, status

bearer_scheme = HTTPBearer()
API_KEY = "changeme-please-set-a-secret"  # move to env

@app.post("/analyze-reply")
async def analyze_reply_endpoint(
    data: dict = Body(...),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    token = credentials.credentials
    if token != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    reply_text = (data.get("reply_text") or "").strip()
    if not reply_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="reply_text required")

    # rest of your analyzer logic...
