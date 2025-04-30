import os
import urllib.request
import zipfile

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download complete!")

def extract_zip(zip_path, extract_path):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete!")

def main():
    # Create scenarios directory if it doesn't exist
    os.makedirs("scenarios", exist_ok=True)
    
    # Download basic scenario
    basic_url = "https://github.com/mwydmuch/ViZDoom/raw/master/scenarios/basic.wad"
    basic_path = os.path.join("scenarios", "basic.wad")
    
    if not os.path.exists(basic_path):
        download_file(basic_url, basic_path)
    
    print("\nAll scenario files downloaded successfully!")

if __name__ == "__main__":
    main() 