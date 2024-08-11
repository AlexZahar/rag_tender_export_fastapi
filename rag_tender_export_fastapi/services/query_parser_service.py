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
Agieren Sie als Vertriebsberater in der Baubranche, der komplexe Ausschreibungstexte versteht. Ihre Aufgabe ist es, spezifische Parameter aus der gegebenen Ausschreibungsbeschreibung zu extrahieren und in einen einzigen String zu formatieren.

Extrahieren Sie alle Eigenschaften des Wandsystems aus dem folgenden Text. Berücksichtigen Sie alle relevanten Details wie System-ID, Abmessungen, Feuerwiderstandsklasse, Materialien und Konstruktionsmerkmale. Entfernen Sie alle Leerzeichen und unnötigen Zeichen, die nicht Teil der Systemeigenschaftsstandards sind.

Formatieren Sie die Ausgabe als einzelnen String in der folgenden Form:
Knauf System ID: [System-ID], Eigenschaften: [Vollständige Beschreibung der Eigenschaften]

Hinweise:
- Die Eigenschaften sollten alle relevanten Details aus dem Eingabetext enthalten.
- Bewahren Sie wichtige Formatierungen, einschließlich Zeilenumbrüche (dargestellt durch </br>).
- Fügen Sie keine Informationen hinzu, die nicht im Originaltext vorhanden sind.
- Entfernen Sie Platzhaltertexte oder Anweisungen (z.B. '(*) Nicht Zutreffendes streichen/ändern/ergänzen').
- Wenn ein spezifischer Wert nicht angegeben ist (z.B. "Wandhöhe: ...... m"), übernehmen Sie ihn so.

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