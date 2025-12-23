import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List


def convert_strokes_to_paths(input_file: str, select_attr: str = 'all') -> bool:
    """
    Convert SVG strokes to paths using Inkscape.

    Args:
        input_file: Path to the input SVG file
        select_attr: CSS selector for elements to convert (default: 'all')

    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Check if Inkscape is installed
        subprocess.run(['inkscape', '--version'],
                     check=True,
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)

        # Determine selection method
        select_method = 'select-all' if select_attr == 'all' else 'select-by-selector'
        print(f"runing Inkscape 'stroke to path' on {input_file}")

        # Build and run the Inkscape command
        cmd = [
            'inkscape',
            f'--actions={select_method}:{select_attr};object-stroke-to-path',
            f'--export-filename={input_file}',
            input_file
        ]

        result = subprocess.run(cmd,
                             check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error converting SVG: {e.stderr.decode()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: Inkscape is not installed or not found in PATH", file=sys.stderr)
        return False


def batch_convert_strokes_to_paths(files: List[str], select_attr: str = 'all') -> List[bool]:
    """
    Convert strokes to paths in multiple SVG files.

    Args:
        files: List of input SVG file paths
        select_attr: CSS selector for elements to convert (default: 'all')

    Returns:
        List of boolean results for each file conversion
    """
    return [convert_strokes_to_paths(file, select_attr) for file in files]


def parallel_convert_strokes_to_paths(files: List[str], select_attr: str = 'all', max_workers: int = 4) -> List[bool]:
    """
    Convert strokes to paths in multiple SVG files in parallel.

    Args:
        files: List of input SVG file paths
        select_attr: CSS selector for elements to convert (default: 'all')
        max_workers: Maximum number of threads to use (default: 4)

    Returns:
        List of boolean results for each file conversion
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_file = {
            executor.submit(convert_strokes_to_paths, file, select_attr): file
            for file in files
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Thread {threading.get_ident()} finished processing {file}")
            except Exception as exc:
                print(f"Thread {threading.get_ident()} generated an exception for {file}: {exc}")
                results.append(False)

    return results