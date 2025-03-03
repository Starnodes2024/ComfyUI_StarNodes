import os
import re

# Define our colors
TITLE_COLOR = "#19124d"  # Dark blue title
BG_COLOR = "#3d124d"     # Purple background
TEXT_COLOR = "#051b34"   # Dark blue text (was white)

# Directory containing all node files
nodes_dir = os.path.dirname(os.path.abspath(__file__))

# Regular expression to find class definitions
class_pattern = re.compile(r'class\s+(\w+)(?:\(.*?\))?:')
# Regular expression to find CATEGORY, COLOR, and BGCOLOR definitions
category_pattern = re.compile(r'\s+CATEGORY\s*=\s*[\'"]([^\'"]+)[\'"]')
color_pattern = re.compile(r'\s+COLOR\s*=\s*[\'"]([^\'"]+)[\'"]')
bgcolor_pattern = re.compile(r'\s+BGCOLOR\s*=\s*[\'"]([^\'"]+)[\'"]')

# Count of files updated
updated_files = 0
updated_nodes = 0

print(f"Scanning directory: {nodes_dir}")

# Process all Python files in the directory
for filename in os.listdir(nodes_dir):
    if not filename.endswith('.py') or filename == '__init__.py' or filename == 'update_node_colors.py':
        continue
    
    filepath = os.path.join(nodes_dir, filename)
    print(f"Processing file: {filename}")
    
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Find all class definitions
    classes = class_pattern.finditer(content)
    file_updated = False
    
    for match in classes:
        class_name = match.group(1)
        class_start = match.start()
        
        # Skip if it's not a node class (no CATEGORY attribute)
        category_match = category_pattern.search(content, class_start)
        if not category_match or not category_match.group(1).startswith('⭐'):
            continue
        
        # Check if COLOR and BGCOLOR are already defined
        color_exists = color_pattern.search(content, class_start)
        bgcolor_exists = bgcolor_pattern.search(content, class_start)
        
        # Prepare the new class attributes
        new_attrs = []
        if not color_exists:
            new_attrs.append(f'    COLOR = "{TITLE_COLOR}"  # Title color')
        else:
            # Replace existing COLOR
            content = re.sub(
                r'(\s+COLOR\s*=\s*)[\'"]([^\'"]+)[\'"]', 
                f'\\1"{TITLE_COLOR}"', 
                content
            )
            
        if not bgcolor_exists:
            new_attrs.append(f'    BGCOLOR = "{BG_COLOR}"  # Background color')
        else:
            # Replace existing BGCOLOR
            content = re.sub(
                r'(\s+BGCOLOR\s*=\s*)[\'"]([^\'"]+)[\'"]', 
                f'\\1"{BG_COLOR}"', 
                content
            )
        
        # If we need to add new attributes
        if new_attrs:
            # Find where to insert the new attributes (after class definition)
            lines = content.split('\n')
            class_line_num = 0
            for i, line in enumerate(lines):
                if class_name in line and 'class' in line:
                    class_line_num = i
                    break
            
            # Insert the new attributes after the class definition
            for attr in new_attrs:
                lines.insert(class_line_num + 1, attr)
            
            content = '\n'.join(lines)
            file_updated = True
            updated_nodes += 1
            print(f"  Updated node: {class_name}")
    
    # Write the updated content back to the file
    if file_updated:
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(content)
        updated_files += 1

print(f"\nUpdate complete! Modified {updated_nodes} nodes in {updated_files} files.")
print(f"Title color: {TITLE_COLOR}")
print(f"Background color: {BG_COLOR}")
print(f"Text color: {TEXT_COLOR}")
