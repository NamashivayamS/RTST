# utils/corrections/proper_noun_corrections.py
"""
Proper noun corrections for Indian names, places, and organizations
that Whisper medium consistently mistranscribes.

Add entries as you discover new mistranscriptions from the terminal logs.
Format: "Whisper's wrong output": "Correct spelling"
"""

PROPER_NOUN_CORRECTIONS = {
    #-----Detected while making the project -----
    "Kureshi": "Accuracy",
    "MEC": "MSc",
    "Iruppur":"Tiruppur",
    "what i am providing is": "Reporting for you, I am",
    "i am providing": "reporting",

    # ── People Names ──────────────────────────────────────────────────────
    "Namah Shivayam": "Namashivayam",
    "Namishwamy": "Namashivayam",
    "Namah Namashivayam": "Namashivayam",
    "Namashwarayam": "Namashivayam",
    "Nameshwarayam": "Namashivayam",
    "Namesh warayam": "Namashivayam",
    "Namashwaram": "Namashivayam",
    "Namashwaramy": "Namashivayam",
    "Namashkayam": "Namashivayam",
    "Namashkayam": "Namashivayam",
    "Namaskaram": "Namashivayam",
    "Namashrayam": "Namashivayam",
    "Namashwayam": "Namashivayam",
    "Namah Shivaya": "Namashivayam",
    "Namesh Roy": "Namashivayam",
    "Namash Royam":"Namashivayam",
    "Navashavyam": "Namashivayam",
    "Nashwayam":"Namashivayam",
    "Namashvayam" :"Namashivayam",
    "Shivayam": "Namashivayam",

    "Thiru valluvar": "Thiruvalluvar",
    "Tiruvaluvar": "Thiruvalluvar",
    "Through valluvar": "Thiruvalluvar",
    "Thiru valar": "Thiruvalluvar",
    "Tiru vallur": "Thiruvalluvar",
    "Thiruvallur": "Thiruvalluvar",
    
    "Bharathiar": "Bharathiyar",
    "Barathiyar": "Bharathiyar",
    "Parathiyar": "Bharathiyar",
    "Bharthiyar": "Bharathiyar",
    "Bharathi yar": "Bharathiyar",

    "Apdul kalam": "Abdul Kalam",
    "Abdool Kalam": "Abdul Kalam",
    "Abdul kallam": "Abdul Kalam",
    "Abdul galam": "Abdul Kalam",

    "Rajnikanth": "Rajinikanth",
    "Rajini kanth": "Rajinikanth",
    "Raginikanth": "Rajinikanth",
    "Rajini kant": "Rajinikanth",

    "Kamal Hassan": "Kamal Haasan",
    "Kamal hasan": "Kamal Haasan",
    "Kamalahasan": "Kamal Haasan",
    "Camel Hassan": "Kamal Haasan",
    "Kamal Hasaan": "Kamal Haasan",

    # ── Places ────────────────────────────────────────────────────────────
    "Lamal": "Lamel",
    "Chelammal": "Chelammal Vidyashram",
    "Chellamal Vidyashram": "Chelammal Vidyashram",
    "Lamal Vidyasram": "Chelammal Vidyashram",

    "Coiambatore": "Coimbatore",
    "Coyambatore": "Coimbatore",
    "Coimpatore": "Coimbatore",
    "Koyambatore": "Coimbatore",
    "Kuyambattur": "Coimbatore",
    "Goyamputhur": "Coimbatore",

    "Mathurai": "Madurai",
    "Madura": "Madurai",
    "Maduray": "Madurai",
    "Mudurai": "Madurai",
    "Mathuray": "Madurai",

    "Tiruchi": "Tiruchirappalli",
    "Thiruchirapalli": "Tiruchirappalli",
    "Tiruchirapali": "Tiruchirappalli",
    "Thiru chirappalli": "Tiruchirappalli",
    "Tiru chirapalli": "Tiruchirappalli",
    "Trichy": "Tiruchirappalli",

    "Kanya kumari": "Kanyakumari",
    "Kanniyakumari": "Kanyakumari",
    "Canyakumari": "Kanyakumari",
    "Kaniyakumari": "Kanyakumari",

    "Tiruvanantapuram": "Thiruvananthapuram",
    "Thiruvandrum": "Thiruvananthapuram",
    "Thiruvanandapuram": "Thiruvananthapuram",
    "Trivandrum": "Thiruvananthapuram",

    "Chennai": "Chennai",
    "Sennai": "Chennai",
    "Schennai": "Chennai",

    # ── Organizations & Brands ────────────────────────────────────────────
    "Ram raj cotton": "Ramraj Cotton",
    "Ramraj cotten": "Ramraj Cotton",
    "Ram raj coten": "Ramraj Cotton",
    "Raamraaj cotton": "Ramraj Cotton",
    "Ram raj": "Ramraj Cotton",
    "Ramroj": "Ramraj",
    "Ramraj Kampani": "Ramraj company",

    "Raamraj":"Ramraj",
    "Ram raj": "Ramraj",

    "Zogo": "Zoho",
    "Soho": "Zoho",
    "Zo ho": "Zoho",
    "Zooho": "Zoho",

    "Fresh works": "Freshworks",
    "Preshworks": "Freshworks",
    "Presh works": "Freshworks",

    "Ana university": "Anna University",
    "Anna univercity": "Anna University",
    "Anna univeristy": "Anna University",
    "Coimbatore Institute of Technology": "Coimbatore Institute of Technology",
}
