# Alphavantage API key
alpha_api_key = "TC95ELSQ7M7YBKV5"

# MongoDB credentials
username = "prod_view_only"
password = "EHS1zakHFIBIrjmL"

# DB and collection names
wall_db_name = "redditSentiments"
comments_coll = "comments_data_and_scores"
tickers_keys = "tickers_keys"
last_upload_time = "last_upload_time"

# DB connection string
mongodb_connection_string = f"mongodb+srv://{username}:{password}@redditsentiments.5bs1x.mongodb.net/{wall_db_name}?retryWrites=true&w=majority"
# mongodb+srv://m001-student:<password>@sandbox.5bs1x.mongodb.net/myFirstDatabase?retryWrites=true&w=majority

# Cut-off date - max date from which to delete comments from the last mention (in hours)
cut_off_hours = 24 * 1

# Filter out comments that have more than specified mentions of different stocks
max_comment_mentions = 10

# Blacklisted tickers
black_listed_keys = [
    "I",
    "ELON",
    "WSB",
    "THE",
    "A",
    "ROPE",
    "YOLO",
    "TOS",
    "CEO",
    "DD",
    "IT",
    "OPEN",
    "ATH",
    "PM",
    "IRS",
    "FOR",
    "DEC",
    "BE",
    "IMO",
    "ALL",
    "RH",
    "EV",
    "TOS",
    "CFO",
    "CTO",
    "DD",
    "BTFD",
    "WSB",
    "OK",
    "PDT",
    "RH",
    "KYS",
    "FD",
    "TYS",
    "US",
    "USA",
    "IT",
    "ATH",
    "RIP",
    "BMW",
    "GDP",
    "OTM",
    "ATM",
    "ITM",
    "IMO",
    "LOL",
    "AM",
    "BE",
    "PR",
    "PRAY",
    "PT",
    "FBI",
    "SEC",
    "GOD",
    "NOT",
    "POS",
    "FOMO",
    "TL;DR",
    "EDIT",
    "STILL",
    "WTF",
    "RAW",
    "PM",
    "LMAO",
    "LMFAO",
    "ROFL",
    "EZ",
    "RED",
    "BEZOS",
    "TICK",
    "IS",
    "PM",
    "LPT",
    "GOAT",
    "FL",
    "CA",
    "IL",
    "MACD",
    "HQ",
    "OP",
    "PS",
    "AH",
    "TL",
    "JAN",
    "FEB",
    "JUL",
    "AUG",
    "SEP",
    "SEPT",
    "OCT",
    "NOV",
    "FDA",
    "IV",
    "ER",
    "IPO",
    "MILF",
    "BUT",
    "SSN",
    "FIFA",
    "USD",
    "CPU",
    "AT",
    "GG",
    "Mar",
    "MOON",
    "HOLD",
    "ON",
    "ARE",
    "DO",
    "GO",
    "B",
    "OC",
    "K",
    "F",
    "F",
    "T",
    "M",
    "U",
    "C",
]

# Streamlit parameters
top_elements = 20
