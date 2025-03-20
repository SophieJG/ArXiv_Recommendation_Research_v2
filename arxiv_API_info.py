import json
import requests
import xml.etree.ElementTree as ET
import sys
from time import sleep

# Base URL for arXiv API
ARXIV_API_BASE = "http://export.arxiv.org/api/query"

def parse_authors(entry):
    authors = []
    for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
        name = author.find("{http://www.w3.org/2005/Atom}name").text
        if name:
            name_parts = name.split(" ")
            last_name = name_parts[-1]
            first_name = " ".join(name_parts[:-1])
            authors.append([last_name, first_name, ""])
    return authors

def fetch_arxiv_data(arxiv_id):
    query = f"id_list={arxiv_id}"
    response = requests.get(f"{ARXIV_API_BASE}?{query}")

    if response.status_code == 200:
        xml_data = response.text
        root = ET.fromstring(xml_data)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")

        if entry is not None:
            paper_id = entry.find("{http://www.w3.org/2005/Atom}id").text.split("/")[-1]
            title = entry.find("{http://www.w3.org/2005/Atom}title").text
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            updated = entry.find("{http://www.w3.org/2005/Atom}updated").text
            authors = ", ".join([author.find("{http://www.w3.org/2005/Atom}name").text for author in entry.findall("{http://www.w3.org/2005/Atom}author")])
            categories = [cat.attrib["term"] for cat in entry.findall("{http://www.w3.org/2005/Atom}category")]
            
            journal_ref = entry.find("{http://arxiv.org/schemas/atom}journal_ref")
            comment = entry.find("{http://arxiv.org/schemas/atom}comment")
            report_number = entry.find("{http://arxiv.org/schemas/atom}report-no")
            doi = entry.find("{http://arxiv.org/schemas/atom}doi")

            journal_ref = journal_ref.text if journal_ref is not None else "N/A"
            comment = comment.text if comment is not None else "N/A"
            report_number = report_number.text if report_number is not None else "N/A"
            doi = doi.text if doi is not None else "N/A"

            versions = []
            for version in entry.findall("{http://arxiv.org/schemas/atom}version"):
                versions.append({
                    "version": version.attrib.get("version", "N/A"),
                    "created": version.text.strip() if version.text else "N/A"
                })

            return {
                "id": paper_id,
                "submitter": "N/A",
                "authors": authors,
                "title": title,
                "comments": comment,
                "journal-ref": journal_ref,
                "doi": doi,
                "report-no": report_number,
                "categories": ", ".join(categories),
                "license": None,
                "abstract": abstract,
                "versions": versions,
                "update_date": updated,
                "authors_parsed": parse_authors(entry)
            }
    return None

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_json_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = "arxiv_papers_formatted.json"

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            arxiv_mapping = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{input_file}' is not a valid JSON file.")
        sys.exit(1)

    arxiv_ids = list(arxiv_mapping.keys())
    arxiv_paper_details = {}

    for arxiv_id in arxiv_ids:
        sleep(1)
        data = fetch_arxiv_data(arxiv_id)
        if data:
            arxiv_paper_details[arxiv_id] = data

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(arxiv_paper_details, f, indent=4)

    print(f"Formatted arXiv paper details saved to '{output_file}'")

if __name__ == "__main__":
    main()