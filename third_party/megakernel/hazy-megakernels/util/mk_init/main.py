#!/usr/bin/env python3
"""
Megakernel Project Initialization Script
Similar to 'npm init', this script sets up a new megakernel project structure.
"""

import sys
import argparse
from pathlib import Path


def prompt_user(question, default=None):
    """Prompt user for input with optional default value."""
    if default:
        response = input(f"{question} ({default}): ").strip()
        return response if response else default
    else:
        response = input(f"{question}: ").strip()
        while not response:
            response = input(f"{question} (required): ").strip()
        return response

def replace_placeholders(content, project_name):
    """Replace placeholders in a string with actual values."""
    content = content.replace('{{PROJECT_NAME_LOWER}}', project_name.lower())
    content = content.replace('{{PROJECT_NAME_UPPER}}', project_name.upper())
    content = content.replace('{{PROJECT_NAME}}', project_name)
    return content

def copy_template_file(source_dir, filename, target_dir, project_name):
    """Copy a template file and replace placeholders."""
    source_file = source_dir / filename
    target_file = target_dir / replace_placeholders(filename, project_name)
    print(f"Copying {source_file} to {target_file}")
    
    if source_file.exists():
        # Read the template file
        with open(source_file, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = replace_placeholders(content, project_name)
        
        # Write to target
        with open(target_file, 'w') as f:
            f.write(content)
        
        print(f"✓ Created {filename}")
    else:
        print(f"⚠ Template file {filename} not found in sources directory")


def create_project_structure(project_name, target_dir):
    """Create the basic project directory structure."""
    directories = [
        'src',
        'tests',
    ]
    
    for directory in directories:
        dir_path = target_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory {directory}/")


def main():
    parser = argparse.ArgumentParser(description='Initialize a new megakernel project')
    parser.add_argument('--name', help='Project name')
    parser.add_argument('--target', help='Target directory (default: current directory)')
    args = parser.parse_args()

    print("🚀 Megakernel Project Initialization")
    print("=" * 40)
    
    # Get project name
    if args.name:
        project_name = args.name
    else:
        project_name = prompt_user("Project name")
    
    # Validate project name
    if not project_name.replace('_', '').replace('-', '').isalnum():
        print("❌ Project name should only contain letters, numbers, hyphens, and underscores")
        sys.exit(1)
    
    # Determine target directory
    if args.target:
        target_dir = Path(args.target).resolve()
    else:
        target_dir = Path.cwd() / project_name
    
    # Check if target directory exists
    if target_dir.exists() and any(target_dir.iterdir()):
        overwrite = input(f"Directory {target_dir} exists and is not empty. Continue? (y/N): ")
        if overwrite.lower() != 'y':
            print("Aborted.")
            sys.exit(1)
    
    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📁 Creating project in: {target_dir}")
    print(f"📦 Project name: {project_name}")
    print()
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    sources_dir = script_dir / "sources"
    
    if not sources_dir.exists():
        print(f"❌ Sources directory not found at {sources_dir}")
        print("Please ensure the template files are available in the sources/ directory")
        sys.exit(1)
    
    # Create project structure
    print("Creating project structure...")
    create_project_structure(project_name, target_dir)
    
    print("\nCopying template files...")
    
    # Copy template files
    template_files = [
        "setup.py",
        "README.md",
        "tests/test_example.py",
        "src/config.cuh",
        "src/{{PROJECT_NAME_LOWER}}.cu",
    ]
    
    for filename in template_files:
        copy_template_file(sources_dir, filename, target_dir, project_name)
    
    # Create a simple .gitignore if it doesn't exist
    gitignore_path = target_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_content = """# Build artifacts
*.o
*.so
*.a
build/
dist/

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Test artifacts
__pycache__/
*.pyc
*.pyo
"""
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("✓ Created .gitignore")
    
    print(f"\n🎉 Successfully initialized megakernel project '{project_name}'!")
    print(f"📍 Location: {target_dir}")
    print("\nNext steps:")
    print(f"  cd {target_dir}")
    print("  make          # Build the project")
    print("  make test     # Run tests")
    print("  make clean    # Clean build artifacts")
    print("\nHappy coding! 🚀")


if __name__ == "__main__":
    main() 