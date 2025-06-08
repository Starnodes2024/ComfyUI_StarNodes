# Star Seven Inputs (txt)

## Description
This node combines up to seven text inputs into a single output string, with a customizable separator. It acts as both an automatic text switch and a concatenation tool, making it versatile for prompt engineering and text processing workflows.

## Inputs
- **separator** (required): The string used to join the text inputs (default: space " ")
- **text1** (optional): First text input (connection point)
- **text2** (optional): Second text input (connection point)
- **text3** (optional): Third text input (connection point)
- **text4** (optional): Fourth text input (text field, default: empty)
- **text5** (optional): Fifth text input (text field, default: empty)
- **text6** (optional): Sixth text input (text field, default: empty)
- **text7** (optional): Seventh text input (text field, default: empty)

## Outputs
- **STRING**: The combined text from all non-empty inputs, joined with the specified separator

## Usage
1. Connect up to three text sources to the first three inputs
2. Enter additional text in the remaining four text fields as needed
3. Specify a separator character or string (default is a space)
4. The node will concatenate all non-empty inputs using the separator

This node is useful for:
- Combining multiple prompt components with consistent formatting
- Creating dynamic text combinations from various sources
- Building complex prompts with both static and dynamic elements
- Automatically using the first available text input (acts as a switch)

If no inputs are provided, the node outputs a default message: "A cute little monster holding a sign with big text: GIVE ME INPUT!"
