import os

# --- Configuration ---
# Add any other folders you want to skip to this set
EXCLUDE_DIRS = {
    'venv',
    '.git',
    '.vscode',
    'node_modules',
    '__pycache__',
    'dist',
    'build',
    '.pytest_cache'
}

# Add any files you want to skip
EXCLUDE_FILES = {
    '.DS_Store',
    '.env'
}
# ---------------------

def get_tree_items(directory):
    """
    Gets a sorted list of items in a directory,
    filtering out excluded ones.
    """
    items = []
    try:
        for item_name in os.listdir(directory):
            path = os.path.join(directory, item_name)
            
            if os.path.isdir(path):
                if item_name not in EXCLUDE_DIRS:
                    items.append(item_name)
            elif os.path.isfile(path):
                if item_name not in EXCLUDE_FILES:
                    items.append(item_name)
                    
        items.sort()  # Sort alphabetically
        return items
        
    except PermissionError:
        return [] # Return an empty list if we can't read the dir

def print_tree(start_dir, prefix=''):
    """
    Recursively prints the directory tree.
    """
    
    items = get_tree_items(start_dir)
    num_items = len(items)

    for i, item_name in enumerate(items):
        path = os.path.join(start_dir, item_name)
        is_last = (i == num_items - 1)

        # Determine the correct markers
        if is_last:
            marker = '\-- '
            new_prefix = prefix + '    '
        else:
            marker = '|-- '
            new_prefix = prefix + '|   '
        
        # Print the item
        print(prefix + marker + item_name)
        
        # If it's a directory, recurse into it
        if os.path.isdir(path):
            print_tree(path, prefix=new_prefix)

# --- Main execution ---
if __name__ == "__main__":
    # Start from the current directory ('.')
    start_path = '.'
    
    # Print the root directory
    print(start_path)
    
    # Start the recursive tree print
    print_tree(start_path)