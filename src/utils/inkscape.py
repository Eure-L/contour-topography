import os
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
        future_to_file = {
            executor.submit(convert_strokes_to_paths, file, select_attr): file
            for file in files
        }

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


def rotate_svg(input_file: str, output_file: str, angle: int) -> bool:
    """
    Rotate an SVG file by a specified angle using Inkscape.

    Args:
        input_file: Path to the input SVG file
        output_file: Path to the output SVG file
        angle: Angle of rotation in degrees (must be a multiple of 90)

    Returns:
        True if rotation was successful, False otherwise
    """
    try:
        # Check if Inkscape is installed
        subprocess.run(['inkscape', '--version'],
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

        # Validate angle
        if angle % 90 != 0:
            raise ValueError("Angle must be a multiple of 90 degrees")

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        cmd = [
            '/usr/bin/inkscape',
            f'--actions',
            f'"select-all;transform-rotate:{angle};export-filename:{input_file};export-do"', f'{input_file}'
        ]

        cmd_str = ' '.join(cmd)
        result = subprocess.run(cmd_str,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error rotating SVG: {e.stderr.decode()}", file=sys.stderr)
        return False
    except FileNotFoundError:
        print("Error: Inkscape is not installed or not found in PATH", file=sys.stderr)
        return False
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return False


def batch_rotate_svg(files: List[str], output_files: List[str], angle: int) -> List[bool]:
    """
    Rotate multiple SVG files by a specified angle using Inkscape.

    Args:
        files: List of input SVG file paths
        output_files: List of output SVG file paths
        angle: Angle of rotation in degrees (must be a multiple of 90)

    Returns:
        List of boolean results for each file rotation
    """
    if len(files) != len(output_files):
        raise ValueError("Input and output file lists must have the same length")

    results = []
    for input_file, output_file in zip(files, output_files):
        result = rotate_svg(input_file, output_file, angle)
        results.append(result)

    return results
