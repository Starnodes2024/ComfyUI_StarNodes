import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import os

class StarNewsScraper:
    """
    A ComfyUI node that scrapes news headlines from a given URL and outputs them as formatted text.
    """
    
    def __init__(self):
        self.sites_file = os.path.join(os.path.dirname(__file__), "sites.txt")
        self.known_sites = self.load_sites()
    
    def load_sites(self) -> List[str]:
        """Load known sites from sites.txt"""
        if not os.path.exists(self.sites_file):
            # Create default sites file with example URL
            default_sites = ["https://news.ycombinator.com"]
            with open(self.sites_file, "w") as f:
                f.write("\n".join(default_sites))
            return default_sites
        
        with open(self.sites_file, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    def save_new_site(self, url: str) -> None:
        """Save a new site to sites.txt if it's not already present"""
        if url not in self.known_sites:
            with open(self.sites_file, "a") as f:
                f.write(f"\n{url}")
            self.known_sites.append(url)
    
    @classmethod
    def INPUT_TYPES(cls):
        instance = cls()
        return {
            "required": {
                "url_choice": (["NEW_URL"] + instance.known_sites,),
                "new_url": ("STRING", {
                    "default": "https://",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "scrape_news"
    CATEGORY = "⭐StarNodes"

    def scrape_headlines(self, url: str) -> List[Dict[str, str]]:
        """
        Scrapes headlines from a given URL.
        """
        try:
            # Fetch the webpage
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find headlines using various common patterns
            # Method 1: Look for common headline tags
            headline_tags = soup.find_all(['h1', 'h2', 'h3', 'h4'])
            
            # Method 2: Look for elements with common headline classes
            headline_classes = soup.find_all(class_=lambda x: x and any(c in str(x).lower() for c in ['title', 'headline', 'heading']))
            
            # Method 3: Look for article titles
            article_titles = soup.find_all('a', class_=lambda x: x and any(c in str(x).lower() for c in ['title', 'headline', 'link']))
            
            # Combine all findings
            all_potential_headlines = headline_tags + headline_classes + article_titles
            
            # Process each headline
            results = []
            seen_headlines = set()  # To avoid duplicates
            
            for tag in all_potential_headlines:
                # Get the headline text and clean it
                headline = tag.get_text().strip()
                
                # Skip if headline is too short or we've seen it before
                if not headline or len(headline) < 10 or headline in seen_headlines:
                    continue
                    
                seen_headlines.add(headline)
                results.append({
                    'headline': headline
                })
                
                # Limit to top 10 headlines to avoid overwhelming output
                if len(results) >= 10:
                    break
            
            return results
        
        except Exception as e:
            print(f"An error occurred while scraping: {str(e)}")
            return []

    def scrape_news(self, url_choice: str, new_url: str):
        """
        Main function that processes the URL and returns formatted text output.
        """
        # Determine which URL to use
        url = new_url if url_choice == "NEW_URL" else url_choice
        
        # Save new URL if it's valid and not already known
        if url_choice == "NEW_URL" and url.startswith(("http://", "https://")):
            self.save_new_site(url)
        
        headlines = self.scrape_headlines(url)
        
        if not headlines:
            return ("No headlines found. The website might be blocking scraping attempts.",)
        
        # Format the output text
        output_text = ""
        
        for item in headlines:
            output_text += f"{item['headline']}.\n"
                    
        return (output_text,)

# This line is required to register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "StarNewsScraper": StarNewsScraper
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "StarNewsScraper": "⭐Star Web Scraper (Headlines) 📰"
}
