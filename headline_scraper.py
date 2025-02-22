import requests
from bs4 import BeautifulSoup
from newspaper import Article
from typing import List, Dict
import sys

def scrape_headlines(url: str) -> List[Dict[str, str]]:
    """
    Scrapes headlines from a given URL and creates summaries.
    
    Args:
        url (str): The URL to scrape headlines from
        
    Returns:
        List[Dict[str, str]]: List of dictionaries containing headlines and their summaries
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
        headlines = []
        
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
            
            # Try to get the article link
            link = None
            if tag.name == 'a':
                link = tag
            else:
                link = tag.find_parent('a') or tag.find('a')
            
            summary = ""
            if link and link.get('href'):
                try:
                    full_url = link['href'] if 'http' in link['href'] else url + link['href']
                    article = Article(full_url)
                    article.download()
                    article.parse()
                    article.nlp()
                    summary = article.summary
                except:
                    summary = "Summary not available"
            
            results.append({
                'headline': headline,
                'summary': summary
            })
            
            # Limit to top 10 headlines to avoid overwhelming output
            if len(results) >= 10:
                break
        
        return results
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return []

def main():
    # Get URL from command line argument or prompt user
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("Enter the URL to scrape headlines from: ")
    
    print(f"\nScraping headlines from: {url}")
    headlines = scrape_headlines(url)
    
    if not headlines:
        print("No headlines found. The website might be blocking scraping attempts.")
        return
    
    print("\nHeadlines and Summaries:")
    print("-" * 50)
    for idx, item in enumerate(headlines, 1):
        print(f"\n{idx}. Headline: {item['headline']}")
        if item['summary']:
            print(f"   Summary: {item['summary'][:200]}...")
        print("-" * 50)

if __name__ == "__main__":
    main()
