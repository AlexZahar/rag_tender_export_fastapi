from llama_index.core import PromptTemplate
from llama_index.llms.openai import OpenAI

query_gen_str_eng = """
Act as a sales advisor in the building industry understanding complex tender text. Your task is to extract specific parameters from the given tender description and format them into a single string.

Extract all the properties of the wall system from the following text. Include all relevant details such as system ID, dimensions, fire resistance class, materials, and construction specifics. Remove any whitespaces and unnecessary characters which are not part of system property standards.

Format the output as a single string in the following form:
System ID: [System ID], Properties: [Full description of properties]

Notes:
- The properties should include all the relevant details provided in the input text.
- Preserve important formatting, including line breaks (represented by </br>).
- Do not add any information that is not present in the original text.
- Remove any placeholder text or instructions (e.g., '(*) Nicht Zutreffendes streichen/ändern/ergänzen').
- If a specific value is not provided (e.g., "Wandhöhe: ...... m"), include it as is.

Input Text:
{query}

Formatted Output:
Generate {num_queries} search query related to the following input query:
Queries:
"""

query_gen_str = """
Bevor Sie die Anfrage analysieren, führen Sie bitte folgende Bereinigungsschritte am Ausschreibungstext durch:

        1. Entfernen Sie alle Zeilenumbrüche und ersetzen Sie sie durch Leerzeichen.
        2. Ersetzen Sie mehrere aufeinanderfolgende Leerzeichen durch ein einzelnes Leerzeichen.
        3. Entfernen Sie Leerzeichen vor und nach Kommas, Punkten, Doppelpunkten und Semikolons.
        4. Entfernen Sie Leerzeichen vor schließenden und nach öffnenden Klammern.
        5. Verbinden Sie getrennte Zahlen (z.B. "12, 5" zu "12,5").
        6. Korrigieren Sie gängige Abkürzungen (z.B. "e. V." zu "e.V.", "z. B." zu "z.B.").
        7. Entfernen Sie Leerzeichen vor % und °.
        8. Stellen Sie sicher, dass nach Satzzeichen ein einzelnes Leerzeichen steht, außer bei Zahlen.
        9. Korrigieren Sie Auslassungspunkte zu "...".
        10. Entfernen Sie Leerzeichen vor Maßeinheiten (mm, m, dB, W, K).
        11. Entfernen Sie Leerzeichen nach 'x' bei Mengenangaben (z.B. "2x 12,5" zu "2x12,5").
        12. Entfernen Sie Leerzeichen vor Sternchen.
        13. Entfernen Sie Leerzeichen um Schrägstriche.
        14. Entfernen Sie führende und nachfolgende Leerzeichen.
        15. Entfernen Sie alle Zeilen, die mit "Orca.Text.ImageRun", "Einheit :" oder "Artikelnr. :" beginnen.
        16. Entfernen Sie die Zeile "(*) Nicht Zutreffendes streichen/ändern/ergänzen".

        Hier ist ein Beispiel für die Anwendung dieser Schritte:

        Originaltext:
        MW12BB, d=125mm
        MW12BB - Metall-Einfachständerwand 2-lagig beplankt, d=125 mm
        als nichttragende innere Trennwand nach DIN 4103-1,
        mit Unterkonstruktion aus verzinkten Stahlblechprofilen mit Oberflächenstruktur, 
        gemäß DIN EN 14195 und DIN 18182-1,
        mit Metallständern CW 75, Boden und Deckenanschlüsse mit Randprofilen UW 75,
        mit beidseitig 2 x 12,5 mm, Schallschutz-Gipsplatten Typ D DIN EN 520 bzw. GKB DIN 18180,
        - Wanddicke: 125 mm,
        - Wandhöhe: ...... m,
        - Befestigungsuntergrund: Stahlbeton/Mauerwerk/ ……………/Leichtbeton (*),
        - Standardverspachtelung Q2 gemäß IGG-Merkblatt 2,
        - Gipsplatten, Spachtel mit Prüfsiegel "geprüft und empfohlen vom IBR"
        System: Rigips MW12BB / Ausführung gemäß Verwendbarkeitsnachweis/Herstellervorschrift,
        Unterkonstruktion:
        mit RigiProfil MultiTec UW / CW 75-06, Ständerabstand 625 mm,
        mit Rigips Anschlussdichtung aus Filz, einseitig selbstklebend,
        Beplankung:
        Beidseitig, 2 x 12,5 mm, Rigips Die Blaue RB,
        mit Rigips Schnellbauschrauben DIN 18182-2 befestigen,
        Verspachtelung:
        Rigips VARIO Fugenspachtel Typ 4B DIN EN 13963,
        Qualitätsstufe Q 2 als Standardverspachtelung, gemäß IGG-Merkblatt 2.
        (*) Nicht Zutreffendes streichen/ändern/ergänzen
        Orca.Text.ImageRun
        Einheit : m²
        Artikelnr. : MW12BB

        Bereinigter Text:
        MW12BB, d=125mm MW12BB - Metall-Einfachständerwand 2-lagig beplankt, d=125mm als nichttragende innere Trennwand nach DIN 4103-1, mit Unterkonstruktion aus verzinkten Stahlblechprofilen mit Oberflächenstruktur, gemäß DIN EN 14195 und DIN 18182-1, mit Metallständern CW 75, Boden und Deckenanschlüsse mit Randprofilen UW 75, mit beidseitig 2x12,5mm, Schallschutz-Gipsplatten Typ D DIN EN 520 bzw. GKB DIN 18180, - Wanddicke: 125mm, - Wandhöhe: ... m, - Befestigungsuntergrund: Stahlbeton/Mauerwerk/.../Leichtbeton(*), - Standardverspachtelung Q2 gemäß IGG-Merkblatt 2, - Gipsplatten, Spachtel mit Prüfsiegel "geprüft und empfohlen vom IBR" System: Rigips MW12BB/Ausführung gemäß Verwendbarkeitsnachweis/Herstellervorschrift, Unterkonstruktion: mit RigiProfil MultiTec UW/CW 75-06, Ständerabstand 625mm, mit Rigips Anschlussdichtung aus Filz, einseitig selbstklebend, Beplankung: Beidseitig, 2x12,5mm, Rigips Die Blaue RB, mit Rigips Schnellbauschrauben DIN 18182-2 befestigen, Verspachtelung: Rigips VARIO Fugenspachtel Typ 4B DIN EN 13963, Qualitätsstufe Q2 als Standardverspachtelung, gemäß IGG-Merkblatt 2.

Eingabetext:
{query}

Formatierte Ausgabe:
Generieren Sie {num_queries} Suchanfragen bezogen auf die folgende Eingabeanfrage:
Anfragen:
"""

query_gen_prompt = PromptTemplate(query_gen_str)

llm = OpenAI(model="gpt-3.5-turbo")


def generate_queries(query: str, llm, num_queries: int = 1):
    response = llm.predict(
        query_gen_prompt, num_queries=num_queries, query=query
    )
    # assume LLM proper put each query on a newline
    queries = response.split("\n")
    queries_str = "\n".join(queries)
    print(f"Generated queries:\n{queries_str}")
    return queries