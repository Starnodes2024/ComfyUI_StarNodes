# Star Text Filter

## Description
The Star Text Filter node provides various text processing capabilities to clean and manipulate string inputs. It can remove text between specified words, filter text before or after specific words, remove empty lines, strip whitespace, and more.

## Inputs
- **text**: The input text string to be filtered or processed
- **filter_type**: The type of filtering to apply:
  - `remove_between_words`: Removes all text between the specified start and end words
  - `remove_before_start_word`: Removes all text before the first occurrence of the start word
  - `remove_after_end_word`: Removes all text after the last occurrence of the end word
  - `remove_empty_lines`: Removes all empty lines from the text
  - `remove_whitespace`: Removes all whitespace characters from the text
  - `strip_lines`: Removes leading and trailing whitespace from each line
- **start_word**: The starting word or phrase for filtering operations (default: "INPUT")
- **end_word**: The ending word or phrase for filtering operations (default: "INPUT")

## Outputs
- **STRING**: The processed text after applying the selected filter

## Usage
1. Connect a text source to the input
2. Select the desired filter type from the dropdown
3. If using word-based filters, specify the start and end words
4. The node will output the filtered text

This node is particularly useful for:
- Cleaning up text prompts
- Extracting specific portions of text
- Formatting text for use in other nodes
- Removing unwanted sections from text inputs
