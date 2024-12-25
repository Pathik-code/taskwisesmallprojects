import os
import requests
from bs4 import BeautifulSoup

# URL of the webpage containing CSV file links
url = "https://www.datablist.com/learn/csv/download-sample-csv-files"

# Directory to save the downloaded CSV files
download_dir = "csv_files"
os.makedirs(download_dir, exist_ok=True)

def download_csv_files(url, download_dir):
    try:
        # Fetch the webpage content
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Find all Google Drive CSV links
        csv_links = soup.find_all("a", href=lambda href: href and "drive.google.com/uc?" in href)
        if not csv_links:
            print("No Google Drive CSV links found!")
            return

        # Download each CSV file
        for link in csv_links:
            file_url = link["href"]
            file_name = link.text.strip()  # Use the text within the <a> tag as the filename
            if not file_name.endswith(".csv"):  # Ensure it ends with .csv
                file_name += ".csv"
            save_path = os.path.join(download_dir, file_name)

            # Download the CSV file
            print(f"Downloading {file_url}...")
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            with open(save_path, "wb") as file:
                file.write(file_response.content)
            print(f"Saved: {save_path}")

        print("All CSV files downloaded successfully!")

    except Exception as e:
        print(f"Error: {e}")

# Run the script
download_csv_files(url, download_dir)                       
