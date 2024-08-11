import ast
import importlib
import os
import sys
from pathlib import Path
import pkg_resources

def get_imports_from_ast(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            tree = ast.parse(file.read())
        except SyntaxError:
            print(f"Syntax error in file: {file_path}")
            return set()
    
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
    return imports

def get_dynamic_imports():
    original_import = __import__
    imported_modules = set()

    def custom_import(name, *args, **kwargs):
        imported_modules.add(name.split('.')[0])
        return original_import(name, *args, **kwargs)

    sys.meta_path.insert(0, custom_import)
    return imported_modules

def run_scripts(directory):
    dynamic_imports = get_dynamic_imports()
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    exec(open(file_path).read())
                except Exception as e:
                    print(f"Error executing {file_path}: {e}")
    return dynamic_imports

def analyze_package_usage(directory):
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    static_imports = set()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                static_imports.update(get_imports_from_ast(file_path))
    
    dynamic_imports = run_scripts(directory)
    
    all_imports = static_imports.union(dynamic_imports)
    
    # Add some commonly used packages that might not be directly imported
    all_imports.update(['pip', 'setuptools', 'wheel'])
    
    # Read requirements from requirements.txt
    req_file = Path(directory) / 'requirements.txt'
    if req_file.exists():
        with req_file.open() as f:
            requirements = [line.strip().split('==')[0].lower() for line in f if line.strip() and not line.startswith('#')]
        installed_packages = set(requirements)
    
    unused_packages = installed_packages - all_imports
    return unused_packages, all_imports

if __name__ == "__main__":
    directory = sys.argv[1] if len(sys.argv) > 1 else '.'
    unused, used = analyze_package_usage(directory)
    
    print("Potentially unused packages:")
    for package in sorted(unused):
        print(f"- {package}")
    
    print("\nUsed packages:")
    for package in sorted(used):
        print(f"- {package}")
    
    print("\nNote: This analysis combines static and dynamic import detection.")
    print("However, it may still miss some runtime dependencies or dynamically loaded modules.")
    print("Please review carefully before removing any packages.")