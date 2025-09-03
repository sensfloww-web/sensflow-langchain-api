from fastapi import FastAPI, Body
import uvicorn
from sensflow_ai import parse_lead, load_properties, match_property, generate_json_reply

app = FastAPI()

@app.post("/process-lead")
def process_lead(data: dict = Body(...)):
    lead_text = data.get("lead_text", "")
    if not lead_text:
        return {"error": "No lead_text provided"}

    try:
        # Step 1: Parse the lead
        lead = parse_lead(lead_text)

        # Step 2: Load property sheet (by Spreadsheet ID + tab index)
        spreadsheet_id = "1Ys5q3g-6fWZK5U2h_tSMVL-ELNqZrIJYI8orJcP6dGE"  # Your sheet ID
        worksheet_index = 3  # Properties tab (0=BuyLead, 1=SellLead, 2=RentLead, 3=Properties)
        df = load_properties(spreadsheet_id, worksheet_index)

        # Step 3: Match properties
        results = match_property(lead, df)

        # Step 4: Generate JSON reply
        reply = generate_json_reply(lead, results)

        return reply

    except Exception as e:
        return {"error": str(e)}

