import urllib.request
import xml.etree.ElementTree as ET
import requests
import pymupdf
from refextract import extract_references_from_string


class PdfDownloader:
    """
    A class to handle downloading PDF files from an API.
    """

    def __init__(self, url: str, file_name: str):
        """
        Initialize the PdfDownloader with a URL and desired file name.

        Args:
            url (str): The URL of the arXiv query to retrieve the XML data from.
            file_name (str): The name of the file to save the downloaded PDF.
        """
        self.url = url
        self.file_name = file_name

    def download_pdf(self):
        xml_data = self._fetch_xml_data()
        pdf_url = self._extract_pdf_url(xml_data)
        self._save_pdf(pdf_url)

    def _fetch_xml_data(self) -> str:
        try:
            data = urllib.request.urlopen(self.url)
            xml_data = data.read().decode("utf-8")
            return xml_data
        except Exception as e:
            raise Exception(f"Error fetching XML data: {e}")

    def _extract_pdf_url(self, xml_data: str) -> str:
        root = ET.fromstring(xml_data)
        pdf_links = [
            link.attrib["href"]
            for link in root.findall(".//{http://www.w3.org/2005/Atom}link")
            if link.attrib.get("title") == "pdf"
        ]
        if not pdf_links:
            raise Exception("No PDF link found in the XML data.")
        return pdf_links[0]

    def _save_pdf(self, pdf_url: str):
        response = requests.get(pdf_url)
        if response.status_code == 200:
            with open(self.file_name, "wb") as file:
                file.write(response.content)
            print(f"Downloaded {self.file_name} successfully!")
        else:
            raise Exception("Failed to download the PDF.")


class PdfReferenceExtractor:
    """
    A class for extracting references from a downloaded PDF.
    """

    def __init__(self, file_name: str):
        """
        Initialize the PdfReferenceExtractor with the path to the PDF file.

        Args:
            file_name (str): The path to the PDF file to extract references from.
        """
        self.file_name = file_name

    def extract_references(self):
        """
        Returns:
            list: A list of dictionaries representing extracted references.
        """
        text = self._extract_text_from_pdf()
        references = extract_references_from_string(text)
        return references

    def _extract_text_from_pdf(self) -> str:
        text = ""
        with pymupdf.open(self.file_name) as doc:
            for page in doc:
                text += page.get_text("text")
        return text


def main():
    url = "http://export.arxiv.org/api/query?search_query=all:electron&start=0&max_results=1"
    file_name = "sample.pdf"

    # Initialize PdfDownloader and download PDF
    downloader = PdfDownloader(url, file_name)
    try:
        downloader.download_pdf()
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        return

    # Initialize PdfReferenceExtractor and extract references
    reference_extractor = PdfReferenceExtractor(file_name)
    try:
        references = reference_extractor.extract_references()
        if references:
            print("Extracted Reference:", references[-1])
        else:
            print("No references found.")
    except Exception as e:
        print(f"Error extracting references: {e}")


if __name__ == "__main__":
    main()
