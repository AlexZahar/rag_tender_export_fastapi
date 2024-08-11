import subprocess
import os

def main():
    script_path = os.path.join(os.path.dirname(__file__), '..', '..', 'start_services.sh')
    subprocess.run(['bash', script_path])

if __name__ == "__main__":
    main()