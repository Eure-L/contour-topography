import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from xml.etree import ElementTree as ET



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
    Rotate an SVG file by a specified angle using Inkscape and update the viewport.

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
            f'"select-all;transform-rotate:{angle};export-filename:{input_file};fit-page-to-drawing;export-do"', f'{input_file}'
        ]

        cmd_str = ' '.join(cmd)
        result = subprocess.run(cmd_str,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Resizes the canevas/Viewport to match the rotation
        tree = ET.parse(input_file)
        root = tree.getroot()

        x0, y0, vb_w, vb_h = root.get('viewBox', '0 0 0 0').split()
        width = root.get('width', '0')
        height = root.get('height', '0')

        # Update viewBox
        root.set('viewBox', f'0 0 {vb_h} {vb_w}')
        root.set('width', height)
        root.set('height', width)
        tree.write(input_file, encoding="utf-8", xml_declaration=True)
        time.sleep(0.5)
        cmd = [
            '/usr/bin/inkscape',
            f'--actions',
            f'"select-all;export-filename:{input_file};selection-move-to-page-center;fit-page-to-drawing;export-do"', f'{input_file}'
        ]


        cmd_str = ' '.join(cmd)
        result = subprocess.run(cmd_str,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Recenters Objects
        # inkscape "--actions=select-all:all;selection-group;object-align:hcenter vcenter page;export-filename:foo.svg;export-do" canberra_400-1900.svg
        cmd = [
            '/usr/bin/inkscape',
            f'--actions',
            f'"select-all:all;selection-group;object-align:hcenter vcenter page;export-filename:{input_file};export-do"', f'{input_file}'
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
    except Exception as e:
        print(f"Error processing SVG: {str(e)}", file=sys.stderr)
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
