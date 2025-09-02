from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

print("✅ Script started")  # Debug

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("🔑 API Key loaded?", bool(api_key))  # Debug

# Setup LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)
print("🤖 LLM initialized")  # Debug

# Define fields to extract
schemas = [
    ResponseSchema(name="intent", description="Lead intent: buy, sell, or rent"),
    ResponseSchema(name="city", description="City of the property"),
    ResponseSchema(name="budget", description="Budget in INR as a number"),
    ResponseSchema(name="bhk", description="Number of bedrooms"),
    ResponseSchema(name="email", description="Lead email address"),
    ResponseSchema(name="phone", description="Lead phone number")
]

parser = StructuredOutputParser.from_response_schemas(schemas)
print("📦 Parser ready")  # Debug

prompt_template = PromptTemplate(
    template="""Extract structured info from this lead message:
    {lead_text}

    {format_instructions}
    """,
    input_variables=["lead_text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def parse_lead(lead_text: str):
    """Parse raw lead text into structured data"""
    print("⚡ Parsing lead:", lead_text)  # Debug
    prompt = prompt_template.format(lead_text=lead_text)
    output = llm.invoke(prompt)
    print("📨 Raw output:", output)  # Debug
    return parser.parse(output.content)

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

def load_properties(sheet_name="Properties"):
    """Load property data from Google Sheet into a Pandas DataFrame"""
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)

    # Use your actual sheet title here 👇
    sheet = client.open("automation").worksheet(sheet_name)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # ✅ Clean the Price column: convert "180k" → 180000, "1.5M" → 1500000
    def parse_price(val):
        if isinstance(val, (int, float)):
            return int(val)
        val = str(val).lower().strip()
        if val.endswith("k"):
            return int(float(val[:-1]) * 1000)
        elif val.endswith("m"):
            return int(float(val[:-1]) * 1000000)
        else:
            try:
                return int(val)
            except:
                return 0

    df["Price"] = df["Price"].apply(parse_price)

    # ✅ Optional: Clean BHK column too ("2BHK" → 2)
    def parse_bhk(val):
        val = str(val).lower().replace("bhk", "").strip()
        try:
            return int(val)
        except:
            return 0

    df["BHK"] = df["BHK"].apply(parse_bhk)

    return df


from difflib import SequenceMatcher

def similar(a, b):
    """Return similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

from difflib import SequenceMatcher

def similar(a, b):
    """Return similarity ratio between two strings"""
    return SequenceMatcher(None, a, b).ratio()

def match_property(lead: dict, df: pd.DataFrame, top_k: int = 3):
    """Smart matching with alternatives (under + above budget)"""
    city = str(lead.get("city", "")).lower()
    budget = int(lead.get("budget", 0))
    bhk = str(lead.get("bhk", "")).replace("BHK", "").strip()

    def matches(row):
        # Fuzzy city match
        city_ok = similar(city, str(row["City"]).lower()) > 0.6 if city else True

        # Budget tolerance (±10%)
        budget_ok = True
        if budget:
            min_budget = budget * 0.9
            max_budget = budget * 1.1
            budget_ok = min_budget <= row["Price"] <= max_budget

        # Flexible BHK (allow ±1)
        bhk_ok = True
        if bhk.isdigit():
            try:
                bhk_val = int(bhk)
                bhk_ok = row["BHK"] in [bhk_val, bhk_val - 1, bhk_val + 1]
            except:
                pass

        return city_ok and budget_ok and bhk_ok

    # --- Step 1: Try exact matches ---
    filtered = df[df.apply(matches, axis=1)]

    if not filtered.empty:
        return {
            "matches": filtered.head(top_k).to_dict(orient="records"),
            "note": "✅ Found exact/smart matches."
        }

    # --- Step 2: If no match, suggest alternatives ---
    alt_under = df[(df["Price"] <= budget)]
    alt_over = df[(df["Price"] > budget) & (df["Price"] <= budget * 1.15)]

    # Combine and sort
    alternatives = pd.concat([alt_under, alt_over]).sort_values(by="Price")

    if not alternatives.empty:
        return {
            "matches": alternatives.head(top_k).to_dict(orient="records"),
            "note": "⚠️ No exact match found. Showing best alternatives under and slightly above your budget."
        }

    # --- Step 3: If still nothing, fallback ---
    fallback = df.sort_values(by="Price", ascending=True).head(top_k)
    return {
        "matches": fallback.to_dict(orient="records"),
        "note": "⚠️ No properties found in your range. Showing closest options."
    }

def generate_reply(lead: dict, results: dict):
    note = results["note"]
    matches = results["matches"]

    email_text = f"Hi {lead.get('email', 'there')},\n\n"
    email_text += note + "\n\n"

    if matches:
        email_text += "Here are the property options we found for you:\n\n"
        for m in matches:
            email_text += (
                f"🏡 {m['Type']} in {m['City']}\n"
                f"📍 Address: {m['Address']}\n"
                f"🛏 BHK: {m['BHK']}\n"
                f"💷 Price: £{int(m['Price']):,}\n"
                f"✨ Features: {m['Features']}\n"
                f"🔗 Link: {m['Link']}\n\n"
            )
    else:
        email_text += "Unfortunately, we couldn't find any properties matching your criteria right now.\n"

    email_text += "Would you like us to expand the search to nearby areas or different budgets?\n\n"
    email_text += "Best regards,\nSensflow Real Estate Assistant"
    return email_text

def generate_json_reply(lead: dict, results: dict):
    """
    Generate JSON response formatted for n8n Email Node with HTML template support.
    """
    reply = {
        "lead_email": lead.get("email", ""),
        "note": results["note"],
        "matches": []
    }

    for m in results["matches"]:
        reply["matches"].append({
            "id": m.get("id", ""),
            "Address": m.get("Address", ""),
            "City": m.get("City", ""),
            "BHK": m.get("BHK", ""),
            "Price": int(m.get("Price", 0)),
            "Type": m.get("Type", ""),
            "Features": m.get("Features", ""),
            "Link": m.get("Link", ""),
            "Image_URL": m.get("Image URL", "")
        })

    return reply

if __name__ == "__main__":
    # Example lead email text
    text = "Hi, I want a 2BHK in England under 200000."
    
    # Step 1: Parse lead
    lead = parse_lead(text)
    print("Parsed Lead:", lead)

    # Step 2: Load property sheet
    df = load_properties("Properties")
    print("Loaded Properties:", df.head())

    # Step 3: Run matching
    results = match_property(lead, df)
    print("Property Match Results:", results)

    # Step 4: Generate reply
    reply = generate_reply(lead, results)
    print("\n--- Generated Reply ---\n")
    print(reply)

    # Step 5: Generate JSON reply
    json_reply = generate_json_reply(lead, results)
    print("\n--- JSON Reply (for n8n) ---\n")
    print(json_reply)
