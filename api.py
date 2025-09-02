from fastapi import FastAPI, Body
import uvicorn
from sensflow_ai import parse_lead, load_properties, match_property, generate_json_reply

app = FastAPI()

@app.post("/process-lead")
def process_lead(data: dict = Body(...)):
    lead_text = data.get("lead_text", "")
    if not lead_text:
        return {"error": "No lead_text provided"}

    # Step 1: Parse the lead
    lead = parse_lead(lead_text)

    # Step 2: Load property sheet
    df = load_properties("Properties")

    # Step 3: Match properties
    results = match_property(lead, df)

    # Step 4: Generate JSON reply
    reply = generate_json_reply(lead, results)

    return reply

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
