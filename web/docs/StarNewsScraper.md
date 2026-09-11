# Star Web Scraper (Headlines)

## Description
The Star Web Scraper is a utility node that extracts headlines from news websites and other web pages. It can scrape content from both predefined websites and custom URLs provided by the user. This node is particularly useful for generating topical prompts, staying updated on current events, or incorporating real-world information into your workflows.

## Inputs

### Required
- **url_choice**: Selection between predefined websites or "NEW_URL" option
- **new_url**: Custom URL to scrape (used when "NEW_URL" is selected)

## Outputs
- **STRING**: Text containing the extracted headlines, one per line

## Usage
1. Select a predefined website from the dropdown list, or choose "NEW_URL" to enter a custom URL
2. If "NEW_URL" is selected, enter the full URL (including http:// or https://) in the "new_url" field
3. Run the node to extract headlines
4. The output text can be connected to text processing nodes or prompt inputs

## Features

### Intelligent Headline Extraction
- Uses multiple methods to identify headlines on different website structures
- Extracts content from common headline tags (h1, h2, h3, h4)
- Identifies elements with headline-related classes
- Finds article titles in links

### Website Management
- Automatically saves new URLs for future use
- Maintains a list of previously used websites
- Stores website URLs in a sites.txt file for persistence between sessions

### Content Processing
- Removes duplicate headlines
- Filters out very short text fragments
- Formats output as clean, readable text
- Limits results to prevent overwhelming outputs

## Technical Details
- Uses requests library for fetching web content
- Implements BeautifulSoup for HTML parsing
- Employs user-agent headers to avoid basic scraping blocks
- Handles errors gracefully with informative messages
- Automatically creates a sites.txt file if one doesn't exist

## Notes
- Some websites may block web scraping attempts
- For best results, use news sites with clearly structured headlines
- The node is limited to extracting up to 10 headlines per request to maintain performance
- Headlines shorter than 10 characters are filtered out
- Custom URLs must include the full protocol (http:// or https://)
- This node requires an internet connection to function
