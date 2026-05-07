import os
import requests
from dotenv import load_dotenv
from datetime import datetime

# ─── Load credentials ────────────────────────────────────
load_dotenv()
USERNAME = os.getenv("NATIONAL_RAIL_USERNAME")
PASSWORD = os.getenv("NATIONAL_RAIL_PASSWORD")

SOAP_URL = "https://ojp.nationalrail.co.uk/webservices"


def format_datetime(date_string: str, hour: int = 9) -> str:
    """
    Parse natural language date/time strings including:
    - "tomorrow", "next Monday", "15th July", "2026-07-15"
    - "tomorrow at 9pm", "Friday 18:30"
    Returns ISO format "YYYY-MM-DDTHH:MM:SS".
    """
    import re as _re
    import dateparser

    # Detect if user specified a time (e.g. "9pm", "18:30")
    has_explicit_time = bool(_re.search(
        r'\d+:\d+|\d+\s*(am|pm)', date_string, _re.IGNORECASE
    ))

    parsed = None
    # Try 1: with PREFER_DATES_FROM=future (handles "tomorrow", "next Monday")
    try:
        parsed = dateparser.parse(
            date_string,
            settings={
                "PREFER_DATES_FROM": "future",
                "RETURN_AS_TIMEZONE_AWARE": False,
                "DATE_ORDER": "DMY",
            }
        )
    except Exception:
        pass

    # Try 2: no settings (handles ISO "2026-07-15" which future-pref can block)
    if not parsed:
        try:
            parsed = dateparser.parse(date_string)
        except Exception:
            pass

    if parsed:
        # Override time with default if user didn't mention one
        if not has_explicit_time:
            parsed = parsed.replace(hour=hour, minute=0, second=0)
        return parsed.strftime("%Y-%m-%dT%H:%M:%S")

    # Fallback: strip ordinals and try strptime
    try:
        clean = date_string.replace("st","").replace("nd","")
        clean = clean.replace("rd","").replace("th","").strip()
        for fmt in ["%d %B %Y", "%d %B", "%B %d", "%Y-%m-%d", "%d/%m/%Y"]:
            try:
                p = datetime.strptime(clean, fmt)
                if p.year == 1900:
                    p = p.replace(year=datetime.now().year)
                return p.strftime(f"%Y-%m-%dT{hour:02d}:00:00")
            except ValueError:
                continue
    except Exception:
        pass
    return datetime.now().strftime(f"%Y-%m-%dT{hour:02d}:00:00")


def build_soap_request(origin_crs, destination_crs, depart_datetime,
                       is_return=False, return_datetime=None):
    return_xml = ""
    if is_return and return_datetime:
        return_xml = f"""
        <ns:inwardTime>
            <ns:departBy>{return_datetime}</ns:departBy>
        </ns:inwardTime>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<soapenv:Envelope
    xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
    xmlns:ns="http://www.thalesgroup.com/ojp/jpservices"
    xmlns:com="http://www.thalesgroup.com/ojp/common">
    <soapenv:Header/>
    <soapenv:Body>
        <ns:RealtimeJourneyPlanRequest>
            <ns:origin>
                <com:stationCRS>{origin_crs}</com:stationCRS>
            </ns:origin>
            <ns:destination>
                <com:stationCRS>{destination_crs}</com:stationCRS>
            </ns:destination>
            <ns:realtimeEnquiry>STANDARD</ns:realtimeEnquiry>
            <ns:outwardTime>
                <ns:departBy>{depart_datetime}</ns:departBy>
            </ns:outwardTime>{return_xml}
            <ns:directTrains>false</ns:directTrains>
            <ns:fareRequestDetails>
                <ns:passengers>
                    <com:adult>1</com:adult>
                    <com:child>0</com:child>
                </ns:passengers>
                <ns:fareClass>STANDARD</ns:fareClass>
            </ns:fareRequestDetails>
            <ns:reducedTransferTime>false</ns:reducedTransferTime>
            <ns:onlySearchForSleeper>false</ns:onlySearchForSleeper>
            <ns:overtakenTrains>false</ns:overtakenTrains>
            <ns:rawRealtimeInfo>false</ns:rawRealtimeInfo>
            <ns:planAnytimeToday>false</ns:planAnytimeToday>
        </ns:RealtimeJourneyPlanRequest>
    </soapenv:Body>
</soapenv:Envelope>"""


def parse_cheapest_fare(xml_text, origin_crs, destination_crs, date):
    date_part = date[:10]
    time_part = date[11:16].replace(":", "") if len(date) > 10 else "0900"
    booking_url = (
        f"https://www.nationalrail.co.uk/journey-planner/"
        f"?from={origin_crs}&to={destination_crs}"
        f"&date={date_part}&time={time_part}&timeoffset=0&type=single"
    )
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)

        NS_COM = "http://www.thalesgroup.com/ojp/common"
        NS_JP  = "http://www.thalesgroup.com/ojp/jpservices"

        response_elem = root.find(f".//{{{NS_COM}}}response")
        if response_elem is not None and response_elem.text != "Ok":
            return {
                "found": False,
                "error": f"API returned: {response_elem.text}",
                "booking_url": booking_url
            }

        cheapest_pence = float("inf")
        cheapest = None

        for journey in root.findall(f".//{{{NS_JP}}}outwardJourney"):
            # Get departure and arrival from timetable
            depart = journey.findtext(
                f".//{{{NS_JP}}}timetable/{{{NS_JP}}}scheduled/{{{NS_JP}}}departure",
                "N/A")
            arrive = journey.findtext(
                f".//{{{NS_JP}}}timetable/{{{NS_JP}}}scheduled/{{{NS_JP}}}arrival",
                "N/A")

            # Fares are under NS_JP not NS_COM
            for fare in journey.findall(f".//{{{NS_JP}}}fare"):
                price_elem = fare.find(f"{{{NS_JP}}}totalPrice")
                desc_elem  = fare.find(f"{{{NS_JP}}}description")

                # Also try NS_COM if NS_JP doesn't work
                if price_elem is None:
                    price_elem = fare.find(f"{{{NS_COM}}}totalPrice")
                    desc_elem  = fare.find(f"{{{NS_COM}}}description")

                if price_elem is not None and price_elem.text:
                    try:
                        pence = int(price_elem.text)
                        if pence < cheapest_pence and pence > 0:
                            cheapest_pence = pence
                            cheapest = {
                                "found": True,
                                "price": f"£{pence/100:.2f}",
                                "ticket_type": desc_elem.text if desc_elem is not None else "Standard",
                                "departure": depart,
                                "arrival": arrive,
                                "booking_url": booking_url
                            }
                    except (ValueError, TypeError):
                        continue

        if cheapest:
            return cheapest

        return {
            "found": False,
            "error": "No fares found in response",
            "booking_url": booking_url,
            "raw_snippet": xml_text[:800]
        }

    except Exception as e:
        return {
            "found": False,
            "error": f"Parse error: {str(e)}",
            "booking_url": booking_url,
            "raw_snippet": xml_text[:800]
        }


def find_cheapest_ticket(origin_crs, destination_crs, date_string,
                          time_string=None, return_date=None):
    travel_date = format_datetime(date_string, hour=9)
    # Build a proper NR journey planner URL with date AND time
    # Format: YYYY-MM-DD for date, HHMM for time
    date_part = travel_date[:10]                    # "2026-07-15"
    time_part = travel_date[11:16].replace(":", "") # "0900"
    booking_url = (
        f"https://www.nationalrail.co.uk/journey-planner/"
        f"?from={origin_crs}&to={destination_crs}"
        f"&date={date_part}&time={time_part}&timeoffset=0&type=single"
    )
    try:
        is_return = return_date is not None
        return_dt = format_datetime(return_date) if return_date else None

        soap_body = build_soap_request(
            origin_crs=origin_crs,
            destination_crs=destination_crs,
            depart_datetime=travel_date,
            is_return=is_return,
            return_datetime=return_dt
        )

        response = requests.post(
            SOAP_URL,
            data=soap_body,
            auth=(USERNAME, PASSWORD),
            headers={"Content-Type": "text/xml; charset=utf-8"},
            timeout=15
        )

        if response.status_code != 200:
            return {
                "found": False,
                "error": f"HTTP {response.status_code}: {response.text[:300]}",
                "booking_url": booking_url
            }

        return parse_cheapest_fare(response.text, origin_crs,
                                    destination_crs, travel_date)

    except requests.exceptions.Timeout:
        return {
            "found": False,
            "error": "API timed out. Try booking directly.",
            "booking_url": booking_url
        }
    except Exception as e:
        return {
            "found": False,
            "error": f"Request error: {str(e)}",
            "booking_url": booking_url
        }


if __name__ == "__main__":
    print("Test 1: Norwich to London Liverpool Street, 15th July")
    result = find_cheapest_ticket("NRW", "LST", "15th July")
    print(f"Found:   {result['found']}")
    if result["found"]:
        print(f"Price:   {result['price']}")
        print(f"Type:    {result['ticket_type']}")
        print(f"Departs: {result['departure']}")
        print(f"Arrives: {result['arrival']}")
    else:
        print(f"Error:   {result.get('error')}")
        if 'raw_snippet' in result:
            print(f"XML:     {result['raw_snippet']}")
    print(f"Book:    {result['booking_url']}")
