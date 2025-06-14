import requests
from bs4 import BeautifulSoup


class DerivativeSource:
    def __init__(self):
        # Base URL for NSE derivatives
        self.url = "https://www.nseindia.com/market-data/derivatives"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        }

    def fetch_data(self):
        """
        Fetch raw HTML data from the source URL.
        """
        try:
            response = requests.get(self.url, headers=self.headers)
            response.raise_for_status()  # Raise an error for HTTP issues
            print("Page fetched successfully.")
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def parse_data(self, html_content):
        """
        Parse the HTML content and extract NSE derivatives data.
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            derivatives_data = []
            table = soup.find("table", {"class": "derivatives-table"})  # Adjust selector
            if table:
                rows = table.find_all("tr")
                for row in rows[1:]:  # Skip the header row
                    cols = [col.text.strip() for col in row.find_all("td")]
                    derivatives_data.append(cols)
            return derivatives_data
        except Exception as e:
            print(f"Error parsing data: {e}")
            return None

    def get_metadata(self):
        """
        Retrieve metadata about the data source.
        """
        return {
            "source_name": "NSE India Derivatives",
            "url": self.url,
            "description": "Fetches derivatives data from NSE India website.",
        }

    def get_data(self):
        """
        High-level method to fetch and parse derivatives data.
        """
        html_content = self.fetch_data()
        if html_content:
            return self.parse_data(html_content)
        return None


# Example usage
if __name__ == "__main__":
    source = DerivativeSource()
    derivatives_data = source.get_data()
    if derivatives_data:
        print("Extracted Derivatives Data:")
        for row in derivatives_data:
            print(row)
    else:
        print("No data extracted.")