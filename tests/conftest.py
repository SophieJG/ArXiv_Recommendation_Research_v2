import os
import sys

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add the src directory to the Python path
sys.path.append(os.path.join(project_root, 'src')) 