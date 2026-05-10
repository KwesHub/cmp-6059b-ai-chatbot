import os
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
USERNAME = os.getenv("NATIONAL_RAIL_USERNAME")
PASSWORD = os.getenv("NATIONAL_RAIL_PASSWORD")

SOAP_URL = "https://ojp.nationalrail.co.uk/webservices"


def format_datetime(date_string: str, hour: int = 9) -> str:
    """Parse natural language date strings like 'tomorrow', '15th July' into ISO format."""
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
                       is_return=False, return_datetime=None,
                       use_first_train=False):
    """
    Build the SOAP XML request body for the National Rail OJP API.
    use_first_train=True searches from the first service of the day,
    which is useful when the user says "any time" and we want the cheapest fare.
    """
    # Extract just the date portion for firstTrainOfDay (xsd:date, not xsd:dateTime)
    date_only = depart_datetime[:10]  # "2026-07-15"

    if use_first_train:
        outward_xml = f"<ns:firstTrainOfDay>{date_only}</ns:firstTrainOfDay>"
    else:
        outward_xml = f"<ns:departBy>{depart_datetime}</ns:departBy>"

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
                {outward_xml}
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


def _nr_booking_url(origin_crs, destination_crs, iso_datetime,
                    ticket_type="single",
                    origin_name=None, destination_name=None,
                    return_datetime=None):
    """
    Build a deep-link to the National Rail OJP timesandfares service.
    Format: /service/timesandfares/{ORIGIN}/{DEST}/{DDMMYY}/{HHMM}/dep
    For returns a second leg is appended: /{RETURN_DDMMYY}/{RETURN_HHMM}/dep
    """
    try:
        dt = datetime.strptime(iso_datetime[:16], "%Y-%m-%dT%H:%M")
        date_str = dt.strftime("%d%m%y")   # e.g. 110526 for 11 May 2026
        time_str = dt.strftime("%H%M")     # e.g. 0900
        url = (
            f"https://ojp.nationalrail.co.uk/service/timesandfares"
            f"/{origin_crs}/{destination_crs}/{date_str}/{time_str}/dep"
        )
        if ticket_type == "return" and return_datetime:
            ret_dt = datetime.strptime(return_datetime[:16], "%Y-%m-%dT%H:%M")
            ret_date = ret_dt.strftime("%d%m%y")
            ret_time = ret_dt.strftime("%H%M")
            url += f"/{ret_date}/{ret_time}/dep"
        return url
    except Exception:
        return "https://ojp.nationalrail.co.uk/service/timesandfares/"


# Keep _ojp_url as an alias for backward compatibility
def _ojp_url(origin_crs, destination_crs, iso_datetime,
             ticket_type="single", origin_name=None, destination_name=None,
             return_datetime=None):
    return _nr_booking_url(origin_crs, destination_crs, iso_datetime,
                           ticket_type, origin_name, destination_name,
                           return_datetime=return_datetime)


def _find_text(elem, *tag_variants):
    """Try multiple XML namespace variants for a tag and return the first match's text."""
    NS_COM = "http://www.thalesgroup.com/ojp/common"
    NS_JP  = "http://www.thalesgroup.com/ojp/jpservices"
    for tag in tag_variants:
        for ns in [NS_JP, NS_COM, ""]:
            key = f"{{{ns}}}{tag}" if ns else tag
            result = elem.findtext(f".//{key}")
            if result is not None:
                return result
    return None


def _find_elem(parent, tag):
    """Find a child element trying both API namespaces and plain tag as fallback."""
    NS_COM = "http://www.thalesgroup.com/ojp/common"
    NS_JP  = "http://www.thalesgroup.com/ojp/jpservices"
    for ns in [NS_JP, NS_COM, ""]:
        key = f"{{{ns}}}{tag}" if ns else tag
        el = parent.find(key)
        if el is not None:
            return el
    return None


_FARE_CATEGORY_LABELS = {
    "ADVANCE":  "Advance",
    "OFF-PEAK": "Off-Peak",
    "ANYTIME":  "Anytime",
}


def parse_cheapest_fare(xml_text, origin_crs, destination_crs, date,
                        booking_url=None):
    if not booking_url:
        booking_url = _nr_booking_url(origin_crs, destination_crs, date)
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml_text)

        NS_COM = "http://www.thalesgroup.com/ojp/common"
        NS_JP  = "http://www.thalesgroup.com/ojp/jpservices"

        # Check API-level response code
        response_elem = root.find(f".//{{{NS_COM}}}response")
        if response_elem is None:
            response_elem = root.find(".//response")
        if response_elem is not None and response_elem.text not in (None, "Ok"):
            return {
                "found": False,
                "error": f"API returned: {response_elem.text}",
                "booking_url": booking_url
            }

        # Collect ALL (pence, departure, arrival, label) tuples
        all_fares = []

        # Try both NS_JP and no-namespace for outwardJourney
        journeys = (root.findall(f".//{{{NS_JP}}}outwardJourney") or
                    root.findall(".//outwardJourney"))

        for journey in journeys:
            depart = _find_text(journey, "departure") or "N/A"
            arrive = _find_text(journey, "arrival") or "N/A"

            fares = (journey.findall(f".//{{{NS_JP}}}fare") or
                     journey.findall(f".//{{{NS_COM}}}fare") or
                     journey.findall(".//fare"))

            for fare in fares:
                price_elem = _find_elem(fare, "totalPrice")
                desc_elem  = _find_elem(fare, "description")
                cat_elem   = _find_elem(fare, "fareCategory")

                if price_elem is not None and price_elem.text:
                    try:
                        pence = int(price_elem.text)
                        if pence <= 0:
                            continue
                        if cat_elem is not None and cat_elem.text:
                            label = _FARE_CATEGORY_LABELS.get(
                                cat_elem.text.upper(), cat_elem.text.title()
                            )
                        elif desc_elem is not None and desc_elem.text:
                            label = desc_elem.text
                        else:
                            label = "Standard"
                        all_fares.append((pence, depart, arrive, label))
                    except (ValueError, TypeError):
                        continue

        if all_fares:
            # Find the minimum price
            min_pence = min(f[0] for f in all_fares)
            # Among all fares at the minimum price, prefer departures at/after 06:00
            cheapest_fares = [f for f in all_fares if f[0] == min_pence]
            chosen = cheapest_fares[0]  # fallback: first (earliest)
            for f in cheapest_fares:
                depart_str = f[1]
                try:
                    # Parse departure time — it's an ISO datetime string
                    depart_hour = int(depart_str[11:13])
                    if depart_hour >= 6:
                        chosen = f
                        break
                except Exception:
                    pass

            pence, depart, arrive, label = chosen
            return {
                "found": True,
                "price": f"£{pence/100:.2f}",
                "ticket_type": label,
                "departure": depart,
                "arrival": arrive,
                "booking_url": booking_url
            }

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


def _single_leg_search(origin_crs, destination_crs, travel_date,
                        use_first_train=False, booking_url=None):
    """Make one SOAP API call for a single leg and return the cheapest fare found."""
    soap_body = build_soap_request(
        origin_crs=origin_crs,
        destination_crs=destination_crs,
        depart_datetime=travel_date,
        is_return=False,
        use_first_train=use_first_train
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
                               destination_crs, travel_date,
                               booking_url=booking_url)


def find_cheapest_ticket(origin_crs, destination_crs, date_string,
                          time_string=None, return_date=None,
                          origin_name=None, destination_name=None,
                          ticket_type="single", use_first_train=False):
    import re as _re
    if _re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', str(date_string)):
        travel_date = date_string
    else:
        travel_date = format_datetime(date_string, hour=9)

    # build the deep-link URL now; return_date is passed through so the return leg
    # date/time appears in the URL for return tickets
    booking_url = _ojp_url(
        origin_crs, destination_crs, travel_date,
        ticket_type=ticket_type,
        origin_name=origin_name,
        destination_name=destination_name,
        return_datetime=return_date if ticket_type == "return" else None
    )

    try:
        if ticket_type == "return" and return_date:
            # Two separate single searches — outward leg + return leg
            # Return leg goes in the opposite direction
            import re as _re2
            if _re2.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}', str(return_date)):
                return_travel_date = return_date
            else:
                return_travel_date = format_datetime(return_date, hour=14)

            use_first_return = return_travel_date.endswith("T00:00:00")

            outward = _single_leg_search(
                origin_crs, destination_crs, travel_date,
                use_first_train=use_first_train, booking_url=booking_url
            )
            inward = _single_leg_search(
                destination_crs, origin_crs, return_travel_date,
                use_first_train=use_first_return, booking_url=booking_url
            )

            if not outward["found"] and not inward["found"]:
                return {"found": False, "error": outward.get("error", "No fares found"),
                        "booking_url": booking_url}

            # Combine prices; fall back gracefully if one leg failed
            out_pence = int(float(outward["price"].replace("£", "")) * 100) if outward["found"] else 0
            in_pence  = int(float(inward["price"].replace("£", "")) * 100) if inward["found"] else 0
            total_pence = out_pence + in_pence

            out_label = outward.get("ticket_type", "Standard") if outward["found"] else "N/A"
            in_label  = inward.get("ticket_type", "Standard") if inward["found"] else "N/A"
            combined_label = out_label if out_label == in_label else f"{out_label} + {in_label}"

            return {
                "found": True,
                "price": f"£{total_pence/100:.2f}",
                "ticket_type": combined_label,
                "departure": outward.get("departure", "N/A") if outward["found"] else "N/A",
                "arrival":   outward.get("arrival",   "N/A") if outward["found"] else "N/A",
                "return_departure": inward.get("departure", "N/A") if inward["found"] else "N/A",
                "return_arrival":   inward.get("arrival",   "N/A") if inward["found"] else "N/A",
                "outward_price": f"£{out_pence/100:.2f}" if outward["found"] else "N/A",
                "inward_price":  f"£{in_pence/100:.2f}"  if inward["found"] else "N/A",
                "booking_url": booking_url
            }

        # Single ticket — one call
        result = _single_leg_search(
            origin_crs, destination_crs, travel_date,
            use_first_train=use_first_train, booking_url=booking_url
        )
        return result

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
    result = find_cheapest_ticket("NRW", "LST", "15th July",
                                   origin_name="Norwich",
                                   destination_name="London Liverpool Street")
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

    print("\nTest 2: firstTrainOfDay (any time)")
    result2 = find_cheapest_ticket("NRW", "LST", "15th July",
                                    origin_name="Norwich",
                                    destination_name="London Liverpool Street",
                                    use_first_train=True)
    print(f"Found:   {result2['found']}")
    if result2["found"]:
        print(f"Price:   {result2['price']}")
        print(f"Type:    {result2['ticket_type']}")
    else:
        print(f"Error:   {result2.get('error')}")
        if 'raw_snippet' in result2:
            print(f"XML:     {result2['raw_snippet']}")
