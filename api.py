from fastapi import FastAPI, Body
import uvicorn
from sensflow_ai import parse_lead, load_properties, match_property, generate_json_reply

app = FastAPI()

@app.post("/process-lead")
def process_lead(data: dict = Body(...)):
    lead_text = data.get("lead_text", "")
    lead_email = data.get("email", "")   # ✅ capture email from request

    if not lead_text:
        return {"error": "No lead_text provided"}

    try:
        # Step 1: Parse the lead
        lead = parse_lead(lead_text)
        if lead_email:   # attach email if provided
            lead["email"] = lead_email

        # Step 2: Load property sheet (by Spreadsheet ID + tab index)
        spreadsheet_id = "1Ys5q3g-6fWZK5U2h_tSMVL-ELNqZrIJYI8orJcP6dGE"  # Your sheet ID
        worksheet_index = 3  # Properties tab
        df = load_properties(spreadsheet_id, worksheet_index)

        # Step 3: Match properties
        results = match_property(lead, df)

        # Step 4: Generate JSON reply
        reply = generate_json_reply(lead, results)

        # ✅ Add intent + email explicitly to response
        reply["intent"] = lead.get("intent", "")
        reply["lead_email"] = lead.get("email", "")

        return reply

    except Exception as e:
        return {"error": str(e)}
@app.post("/process-lead")
def process_lead(data: dict = Body(...)):
    lead_text = data.get("lead_text", "")
    lead_email = data.get("email", "")   # ✅ capture email from request

    if not lead_text:
        return {"error": "No lead_text provided"}

    try:
        # Step 1: Parse the lead
        lead = parse_lead(lead_text)
        if lead_email:   # attach email if provided
            lead["email"] = lead_email

        # Step 2: Load property sheet (by Spreadsheet ID + tab index)
        spreadsheet_id = "1Ys5q3g-6fWZK5U2h_tSMVL-ELNqZrIJYI8orJcP6dGE"
        worksheet_index = 3
        df = load_properties(spreadsheet_id, worksheet_index)

        # Step 3: Match properties
        results = match_property(lead, df)

        # Step 4: Generate JSON reply
        reply = generate_json_reply(lead, results)

        # ✅ Add intent + email explicitly to response
        reply["intent"] = lead.get("intent", "")
        reply["lead_email"] = lead.get("email", "")

        return reply

    except Exception as e:
        return {"error": str(e)}

# 👇👇 paste new endpoints AFTER this line 👇👇

@app.post("/generate-followup")
def generate_followup_endpoint(data: dict = Body(...)):
    ...

@app.post("/analyze-reply")
def analyze_reply_endpoint(data: dict = Body(...)):
    ...
