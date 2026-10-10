"""Dataset schema, validation, and reproducible synthetic demonstration data."""
from __future__ import annotations

from io import BytesIO
import pandas as pd

TEXT_COLUMN = "complaint_text"
CATEGORY_COLUMN = "category"
SENTIMENT_COLUMN = "sentiment"
PRIORITY_COLUMN = "priority"
REQUIRED_COLUMNS = [TEXT_COLUMN, CATEGORY_COLUMN, SENTIMENT_COLUMN, PRIORITY_COLUMN]
CATEGORIES = ["Delay", "Overcrowding", "Driver/Staff Behaviour", "Cleanliness", "Route/Stop Issue", "Fare/Ticketing", "Safety", "Other"]
SENTIMENTS = ["Positive", "Neutral", "Negative"]
PRIORITIES = ["Low", "Medium", "High"]

_SYNTHETIC_RECORDS: dict[str, list[tuple[str, str, str]]] = {
    "Delay": [
        ("The bus was delayed by 30 minutes at Swargate during morning peak hours", "Negative", "High"),
        ("Bus service arrived 15 minutes late at the Deccan bus stop", "Negative", "Medium"),
        ("Long delay waiting because the scheduled bus did not arrive on time", "Negative", "High"),
        ("The bus was delayed by 25 minutes due to heavy traffic on Katraj road", "Negative", "Medium"),
        ("Bus schedule was delayed by 40 minutes leaving passengers waiting at shelter", "Negative", "High"),
        ("Morning bus was delayed by 20 minutes causing passengers to miss connection", "Negative", "High"),
        ("Bus arrived late by 10 minutes at Hadapsar due to signal failure", "Neutral", "Medium"),
        ("Significant delay of 35 minutes on the Hinjewadi bus route", "Negative", "High"),
        ("Evening transit bus was delayed by 15 minutes at Pune Station", "Negative", "Medium"),
        ("Frequent bus delays on route 111 disrupt daily commute for passengers", "Negative", "High"),
        ("Bus arrived on time today at Shivajinagar and the trip was very smooth", "Positive", "Low"),
        ("Timely bus arrival on the BRT corridor with zero delay reported", "Positive", "Low"),
        ("The scheduled bus arrived promptly on time for the morning trip", "Positive", "Low"),
        ("Punctual bus departure from depot made the journey convenient", "Positive", "Low"),
        ("Bus came on time today and reached destination without delay", "Positive", "Low"),
        ("Minor 5 minute delay at Nal stop but driver reached on time", "Neutral", "Low"),
        ("Schedule timing delayed by 15 minutes for the Kothrud depot service", "Negative", "Medium"),
        ("Expected bus delayed by 30 minutes without any announcement", "Negative", "High"),
        ("Bus missed the scheduled arrival time by 20 minutes at Corporation stop", "Negative", "Medium"),
        ("Bus arrived 12 minutes late at terminal due to road repair works", "Negative", "Medium"),
    ],
    "Overcrowding": [
        ("The bus was completely packed and overcrowded so passengers could not board", "Negative", "High"),
        ("Too many people standing in the overcrowded bus with barely space to move", "Negative", "High"),
        ("Severe overcrowding on the bus with passengers dangerously on footboard", "Negative", "High"),
        ("The bus was packed to full capacity leaving commuters stranded at stop", "Negative", "High"),
        ("Heavy rush and overcrowding inside the bus made reaching exit difficult", "Negative", "Medium"),
        ("Overcrowded bus during rush hour with passengers pushed near the doors", "Negative", "High"),
        ("Bus was completely full and overcrowded bypassing waiting commuters", "Negative", "High"),
        ("Excessive passenger crowd on the bus requires urgent extra service", "Negative", "High"),
        ("Dangerous overcrowding inside the bus prevented doors from closing properly", "Negative", "High"),
        ("Morning bus was excessively crowded and packed like sardines", "Negative", "High"),
        ("There was plenty of room in the bus and boarding was very comfortable", "Positive", "Low"),
        ("Spacious bus interior with ample vacant seats and peaceful commute", "Positive", "Low"),
        ("Bus had light passenger crowd and everyone found a comfortable seat", "Positive", "Low"),
        ("Enjoyed a relaxing ride with no overcrowding and plenty of open seats", "Positive", "Low"),
        ("Comfortable journey with low crowd and easy boarding at Shivajinagar", "Positive", "Low"),
        ("Moderate crowd inside the bus but managed well with enough space", "Neutral", "Low"),
        ("Standard passenger occupancy on the bus during afternoon hours", "Neutral", "Low"),
        ("Overcrowding in the bus aisle made moving around uncomfortable", "Negative", "Medium"),
        ("Crowd levels were normal and seating was easily available on bus", "Neutral", "Low"),
        ("Bus was moderately crowded during evening rush but bearable", "Neutral", "Medium"),
    ],
    "Driver/Staff Behaviour": [
        ("The driver spoke extremely rudely when asked about the bus stop", "Negative", "High"),
        ("The conductor was very rude and shouted aggressively at a passenger", "Negative", "High"),
        ("Conductor ignored my polite question about the ticket fare with rude attitude", "Negative", "Medium"),
        ("Driver behaved aggressively and used harsh disrespectful language", "Negative", "High"),
        ("Staff member at bus enquiry counter was dismissive and extremely rude", "Negative", "Medium"),
        ("The driver refused to stop and spoke arrogantly to waiting passengers", "Negative", "High"),
        ("Conductor was arguing rudely with commuters over loose coin change", "Negative", "High"),
        ("Unprofessional behavior by conductor who insulted an elderly commuter", "Negative", "High"),
        ("Staff member showed rude behavior and refused to assist with directions", "Negative", "Medium"),
        ("Driver slammed the door on a passenger with hostile and rude conduct", "Negative", "High"),
        ("The driver was very polite, courteous, and waited for running passengers", "Positive", "Low"),
        ("Conductor was remarkably helpful and patiently assisted an elderly person", "Positive", "Low"),
        ("Staff member kindly guided commuters and answered questions with respect", "Positive", "Low"),
        ("Friendly conductor greeted commuters and issued tickets very politely", "Positive", "Low"),
        ("Driver demonstrated exemplary patience and polite behavior throughout", "Positive", "Low"),
        ("Staff assisted a disabled passenger with great empathy and politeness", "Positive", "Low"),
        ("Courteous staff behavior made the journey pleasant and stress free", "Positive", "Low"),
        ("Conductor politely explained the route map and transit stages", "Positive", "Low"),
        ("Driver maintained a calm, respectful demeanor despite heavy traffic", "Positive", "Low"),
        ("Helpful crew ensured smooth communication and polite assistance", "Positive", "Low"),
    ],
    "Cleanliness": [
        ("The bus floor was dirty and the seats had rubbish and litter everywhere", "Negative", "High"),
        ("Foul smell and dirty condition inside the poorly maintained bus", "Negative", "High"),
        ("The bus was dirty with food waste and empty wrappers on the floor", "Negative", "Medium"),
        ("Seats were covered with dirt and dust making clothes soiled", "Negative", "Medium"),
        ("Tobacco stains and filthy grab handles inside the uncleaned bus", "Negative", "High"),
        ("Disgusting garbage and dirty trash piled under the passenger seats", "Negative", "High"),
        ("The bus interior was muddy, dirty, and smelled of foul garbage", "Negative", "High"),
        ("Sticky dirty handrails and unwashed windows inside the old bus", "Negative", "Medium"),
        ("Cockroaches and dirt spotted inside the unhygienic dirty vehicle", "Negative", "High"),
        ("Seats were dirty and stained with unpleasant odor throughout bus", "Negative", "Medium"),
        ("The vehicle was clean, spotless, and comfortable to travel in today", "Positive", "Low"),
        ("Immaculately clean and sanitized bus interior with neat seats", "Positive", "Low"),
        ("Bus was washed cleanly with spotless floor and tidy seating area", "Positive", "Low"),
        ("Very hygienic clean bus with polished handrails and pleasant smell", "Positive", "Low"),
        ("Fresh and clean vehicle interior made the morning commute pleasant", "Positive", "Low"),
        ("Appreciate the high cleanliness standards and spotless bus seats", "Positive", "Low"),
        ("Tidy and sanitized bus with dust free seats and clean aisle", "Positive", "Low"),
        ("Clean and comfortable passenger cabin maintained by depot staff", "Positive", "Low"),
        ("Neat and well maintained clean seating environment on the new bus", "Positive", "Low"),
        ("Impressed by the vehicle cleanliness and fresh fragrance inside", "Positive", "Low"),
    ],
    "Route/Stop Issue": [
        ("The bus skipped my scheduled stop at Deccan without stopping", "Negative", "High"),
        ("The route display board on the bus showed the incorrect destination", "Negative", "Medium"),
        ("Bus took an unexpected route diversion and bypassed scheduled stops", "Negative", "High"),
        ("Driver did not halt at designated bus stop despite passengers signaling", "Negative", "High"),
        ("The bus bypassed three major route stops leaving commuters stranded", "Negative", "High"),
        ("Front route board did not match the actual route the bus was running on", "Negative", "Medium"),
        ("Driver skipped the Swargate stop forcing passengers to walk back", "Negative", "High"),
        ("Bus route display was blank and passengers were confused about stops", "Negative", "Medium"),
        ("Unauthorized route shortcut taken by driver bypassing passenger stops", "Negative", "High"),
        ("Driver refused to halt at alight stop despite passenger bell ringing", "Negative", "High"),
        ("The bus stop was clearly marked with accurate route timetable board", "Positive", "Low"),
        ("Clear electronic route display and prompt announcements for each stop", "Positive", "Low"),
        ("Bus followed the exact scheduled route and stopped at every stop", "Positive", "Low"),
        ("Proper route navigation with all designated transit stops served well", "Positive", "Low"),
        ("Accurate route indicators made identifying the right bus stop easy", "Positive", "Low"),
        ("Well marked bus stop shelter with clear route map information", "Positive", "Low"),
        ("Bus arrived at the correct stop with clear destination signage", "Positive", "Low"),
        ("Please add a permanent bus stop near the hospital on this route", "Neutral", "Medium"),
        ("Kindly provide an additional request stop on the university route", "Neutral", "Low"),
        ("Query regarding the new route number and intermediate stop stages", "Neutral", "Low"),
    ],
    "Fare/Ticketing": [
        ("The ticket machine charged me twice on UPI for a single bus ticket", "Negative", "High"),
        ("Conductor gave wrong change for ticket and refused to return balance", "Negative", "High"),
        ("Electronic ticketing machine failed and conductor gave no ticket receipt", "Negative", "High"),
        ("Conductor refused online payment and demanded cash for the bus ticket", "Negative", "Medium"),
        ("Overcharged passenger by demanding ten rupees above official ticket fare", "Negative", "High"),
        ("Smart card pass scanner deducted double ticket fare on single tap", "Negative", "High"),
        ("Ticket dispenser device broke down delaying ticket issuance for commuters", "Negative", "Medium"),
        ("Conductor charged adult ticket fare for student with valid travel pass", "Negative", "Medium"),
        ("Dispute over bus ticket fare because conductor pocketed remaining change", "Negative", "High"),
        ("Ticket machine ran out of paper roll and passengers traveled without ticket", "Negative", "Medium"),
        ("The fare was clear and digital QR payment for ticket worked smoothly", "Positive", "Low"),
        ("Transparent ticket fare deduction with instant digital paper ticket", "Positive", "Low"),
        ("Quick ticket issuance by conductor and exact balance returned promptly", "Positive", "Low"),
        ("Seamless ticketing experience using mobile app and digital pass reader", "Positive", "Low"),
        ("Convenient smart card tap payment for ticket at the entrance door", "Positive", "Low"),
        ("Clear ticket printout with exact fare details and stage distance", "Positive", "Low"),
        ("Fast and courteous ticket issuance with proper fare change returned", "Positive", "Low"),
        ("Please clarify revised bus ticket fare rates and monthly pass prices", "Neutral", "Low"),
        ("Inquiry regarding student concession fare and daily ticket rules", "Neutral", "Low"),
        ("General question about ticket pricing for children on AC bus route", "Neutral", "Low"),
    ],
    "Safety": [
        ("The bus was driven dangerously fast through traffic and jumped red lights", "Negative", "High"),
        ("I felt unsafe because the bus door stayed wide open while moving at speed", "Negative", "High"),
        ("Driver was speeding recklessly near Katraj ghat endangering passengers", "Negative", "High"),
        ("A passenger slipped and fell because driver accelerated before boarding", "Negative", "High"),
        ("Bus brakes were failing and screeching loudly creating extreme panic", "Negative", "High"),
        ("Driver was using mobile phone while steering the speeding crowded bus", "Negative", "High"),
        ("Broken handrail near the door is loose and presents severe safety hazard", "Negative", "High"),
        ("Emergency exit door on the bus was jammed and locked shut", "Negative", "High"),
        ("Rash dangerous driving on the wrong side of the road by the bus driver", "Negative", "High"),
        ("Severely worn out bald tyres on the bus pose severe safety accident risk", "Negative", "High"),
        ("The driver followed all traffic safety rules carefully and I felt safe", "Positive", "Low"),
        ("Disciplined safe driving with smooth braking and alert precautions", "Positive", "Low"),
        ("Careful defensive driving maintained during heavy monsoon rainstorm", "Positive", "Low"),
        ("Safety instructions were clear and emergency doors were fully functional", "Positive", "Low"),
        ("Reassuring and safe journey with driver adhering to speed restrictions", "Positive", "Low"),
        ("Smooth and safe ride with excellent road discipline by the driver", "Positive", "Low"),
        ("Pleased with the high safety standards and cautious transit driving", "Positive", "Low"),
        ("First aid safety kit was properly inspected and available on board", "Positive", "Low"),
        ("Side mirror was broken leaving a dangerous blind spot for safety", "Negative", "Medium"),
        ("Minor jerk during sudden braking at pedestrian crossing", "Neutral", "Medium"),
    ],
    "Other": [
        ("I would like information about lost property left on the morning bus", "Neutral", "Medium"),
        ("Left my bag on route 11 bus, where is the central lost property office", "Neutral", "Medium"),
        ("Please share an official update regarding monthly pass renewal counter", "Neutral", "Low"),
        ("Forgot spectacles case on the bus, kindly guide me to lost items desk", "Neutral", "Low"),
        ("I have a general inquiry about weekend transit timings and timetables", "Neutral", "Low"),
        ("Request customer care helpline contact details for Pune transit queries", "Neutral", "Low"),
        ("General feedback regarding the user interface of transit mobile app", "Neutral", "Low"),
        ("Question regarding luggage allowance rules and baggage weight limits", "Neutral", "Low"),
        ("Lost important office files on the bus, please help track lost folder", "Neutral", "High"),
        ("Where can I submit a general enquiry regarding senior citizen bus pass", "Neutral", "Low"),
        ("Thank you for the helpful, reliable, and convenient transit service", "Positive", "Low"),
        ("The journey was pleasant, convenient, and comfortable throughout", "Positive", "Low"),
        ("Commending Pune transit for introducing modern electric feeder buses", "Positive", "Low"),
        ("Appreciation for the prompt response from customer support team", "Positive", "Low"),
        ("Great initiative in improving public transport connectivity in the city", "Positive", "Low"),
        ("Grateful for the dedicated bus services that make daily travel easy", "Positive", "Low"),
        ("Positive remarks on overall transit efficiency and affordable commute", "Positive", "Low"),
        ("Dependable transit network supporting thousands of daily passengers", "Positive", "Low"),
        ("Overall satisfactory travel experience with city bus transport system", "Positive", "Low"),
        ("Appreciating the reliable and prompt bus services across the city", "Positive", "Low"),
    ],
}


def build_synthetic_dataset() -> pd.DataFrame:
    """Create a balanced, realistic, and reproducible transit dataset with authentic labels."""
    rows = []
    for category in CATEGORIES:
        records = _SYNTHETIC_RECORDS.get(category, [])
        for text, sentiment, priority in records:
            rows.append({
                TEXT_COLUMN: text,
                CATEGORY_COLUMN: category,
                SENTIMENT_COLUMN: sentiment,
                PRIORITY_COLUMN: priority,
            })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)



def validate_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, int]]:
    """Return cleaned rows, actionable errors, and validation counts."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        return pd.DataFrame(columns=REQUIRED_COLUMNS), ["Missing required columns: " + ", ".join(missing)], {"input_rows": len(frame), "valid_rows": 0, "empty_text_rows": 0, "duplicate_rows": 0, "unknown_label_rows": 0}
    data = frame[REQUIRED_COLUMNS].copy()
    for c in REQUIRED_COLUMNS:
        data[c] = data[c].astype("string").str.strip()
    empty_mask = data[TEXT_COLUMN].isna() | data[TEXT_COLUMN].eq("")
    empty_count = int(empty_mask.sum())
    duplicate_mask = data.duplicated(subset=REQUIRED_COLUMNS, keep="first")
    duplicate_count = int(duplicate_mask.sum())
    known = data[CATEGORY_COLUMN].isin(CATEGORIES) & data[SENTIMENT_COLUMN].isin(SENTIMENTS) & data[PRIORITY_COLUMN].isin(PRIORITIES)
    unknown_count = int((~known & ~empty_mask).sum())
    if empty_count: errors.append(f"{empty_count} row(s) have missing or empty complaint text.")
    if duplicate_count: errors.append(f"{duplicate_count} exact duplicate row(s) were removed.")
    if unknown_count: errors.append(f"{unknown_count} row(s) contain unknown category, sentiment, or priority labels.")
    valid = data.loc[~empty_mask & known & ~duplicate_mask].copy()
    return valid, errors, {"input_rows": len(frame), "valid_rows": len(valid), "empty_text_rows": empty_count, "duplicate_rows": duplicate_count, "unknown_label_rows": unknown_count}


def read_uploaded_csv(raw: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes, including UTF-8 BOM files."""
    return pd.read_csv(BytesIO(raw), encoding="utf-8-sig")


def prepare_batch_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Validate a prediction upload and drop only rows without complaint text."""
    if TEXT_COLUMN not in frame.columns:
        raise ValueError(f"CSV must contain a `{TEXT_COLUMN}` column.")
    result = frame.copy()
    text = result[TEXT_COLUMN].fillna("").astype(str).str.strip()
    empty = text.eq("")
    result = result.loc[~empty].copy()
    result[TEXT_COLUMN] = text.loc[~empty]
    return result, int(empty.sum())
