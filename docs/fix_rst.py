"""Fix escaped underscores in sphinx-apidoc generated RST files."""

import re
from pathlib import Path


def fix_rst_file(filepath):
    """Remove backslash escapes from automodule directives."""
    content = filepath.read_text()

    # Fix pattern: .. automodule:: module\_name -> .. automodule:: module_name
    fixed = re.sub(
        r'(\.\. automodule:: [^\s]*)\\_',
        lambda m: m.group(1).replace('\\_', '_'),
        content
    )

    # Fix pattern for all escaped underscores in automodule lines
    fixed = re.sub(
        r'(\.\. auto(?:module|class|function):: .*?)\n',
        lambda m: m.group(1).replace('\\_', '_') + '\n',
        fixed
    )

    if fixed != content:
        filepath.write_text(fixed)
        print(f"Fixed: {filepath}")
        return True
    return False


def main():
    api_dir = Path(__file__).parent / 'source' / 'api'
    if not api_dir.exists():
        print(f"API directory not found: {api_dir}")
        return

    fixed_count = 0
    for rst_file in api_dir.rglob('*.rst'):
        if fix_rst_file(rst_file):
            fixed_count += 1

    print(f"Fixed {fixed_count} files")


if __name__ == '__main__':
    main()