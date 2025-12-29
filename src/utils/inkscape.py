import logging
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
from xml.etree import ElementTree as ET

logger = logging.getLogger()


def convert_strokes_to_paths_for_selectors(input_file: str, select_attrs: List[str]) -> bool:
    """
    Convert SVG strokes to paths for multiple selectors using Inkscape.

    :param input_file: Path to the input SVG file
    :param select_attrs: List of CSS selectors for elements to convert

    :return: True if conversion was successful, False otherwise
    """
    start_time = time.time()
    try:
        # Check if Inkscape is installed
        subprocess.run(['inkscape', '--version'],
                       check=True,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)

        actions = []
        for selector in select_attrs:
            actions.append(f'select-by-selector:{selector};object-stroke-to-path')

        actions_str = ';'.join(actions)

        cmd = [
            'inkscape',
            f'--actions={actions_str}',
            f'--export-filename={input_file}',
            input_file
        ]

        logger.debug(' '.join(cmd))
        result = subprocess.run(cmd,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        duration = time.time() - start_time
        logger.debug(f"Stroke conversion for {input_file} completed in {duration:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error converting SVG: {e.stderr.decode()}", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Stroke conversion failed for {input_file} after {duration:.2f} seconds: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: Inkscape is not installed or not found in PATH", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Stroke conversion failed for {input_file} after {duration:.2f} seconds: Inkscape not found")
        return False


def parallel_convert_strokes_to_paths(files: List[str], select_attrs: List[str] = ['[type="road"]'],
                                      max_workers: int = 4) -> List[bool]:
    """
    Convert strokes to paths in multiple SVG files in parallel for multiple element types.

    :param files: List of input SVG file paths
    :param select_attrs: List of CSS selectors for elements to convert (default: ['[type="road"]'])
    :param max_workers: Maximum number of threads to use (default: 4)

    :return: List of boolean results for each file conversion
    """
    total_start_time = time.time()
    results = []
    thread_times = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(convert_strokes_to_paths_for_selectors, file, select_attrs): file
            for file in files
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            thread_id = threading.get_ident()
            thread_start_time = time.time()

            try:
                result = future.result()
                results.append(result)
                duration = time.time() - thread_start_time
                thread_times[thread_id] = duration
                logger.debug(f"Finished 'stroke to path' -> {file} in {duration:.2f} seconds")

            except Exception as exc:
                duration = time.time() - thread_start_time
                logger.warning(
                    f"Thread {thread_id} generated an exception for {file} after {duration:.2f} seconds: {exc}")
                thread_times[thread_id] = duration
                results.append(False)

    total_duration = time.time() - total_start_time
    logger.debug(f"Parallel stroke conversion completed in {total_duration:.2f} seconds")

    return results


def rotate_svg(input_file: str, output_file: str, angle: int) -> bool:
    """
    Rotate an SVG file by a specified angle using Inkscape and update the viewport.

    :param input_file: Path to the input SVG file
    :param output_file: Path to the output SVG file
    :param angle: Angle of rotation in degrees (must be a multiple of 90)

    :return: True if rotation was successful, False otherwise
    """
    start_time = time.time()
    try:
        logger.debug(f"Starting Inkscape {angle}° rotation -> {input_file}")

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
            f'"select-all;transform-rotate:{angle};export-filename:{input_file};fit-page-to-drawing;export-do"',
            f'{input_file}'
        ]

        cmd_str = ' '.join(cmd)
        logger.debug(cmd_str)
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
            f'"select-all;export-filename:{input_file};selection-move-to-page-center;fit-page-to-drawing;export-do"',
            f'{input_file}'
        ]

        cmd_str = ' '.join(cmd)
        result = subprocess.run(cmd_str,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # Recenters Objects
        cmd = [
            '/usr/bin/inkscape',
            f'--actions',
            f'"select-all:all;selection-group;object-align:hcenter vcenter page;export-filename:{input_file};export-do"',
            f'{input_file}'
        ]

        cmd_str = ' '.join(cmd)
        result = subprocess.run(cmd_str,
                                check=True,
                                shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        duration = time.time() - start_time
        logger.debug(f"Rotation for {input_file} completed in {duration:.2f} seconds")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Error rotating SVG: {e.stderr.decode()}", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Rotation failed for {input_file} after {duration:.2f} seconds: {e.stderr.decode()}")
        return False
    except FileNotFoundError:
        print("Error: Inkscape is not installed or not found in PATH", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Rotation failed for {input_file} after {duration:.2f} seconds: Inkscape not found")
        return False
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Rotation failed for {input_file} after {duration:.2f} seconds: {e}")
        return False
    except Exception as e:
        print(f"Error processing SVG: {str(e)}", file=sys.stderr)
        duration = time.time() - start_time
        logger.error(f"Rotation failed for {input_file} after {duration:.2f} seconds: {str(e)}")
        return False


def batch_rotate_svg(files: List[str], output_files: List[str], angle: int, max_workers: int = 4) -> List[bool]:
    """
    Rotate multiple SVG files by a specified angle using Inkscape in parallel.

    :param files: List of input SVG file paths
    :param output_files: List of output SVG file paths
    :param angle: Angle of rotation in degrees (must be a multiple of 90)
    :param max_workers: Maximum number of threads to use (default: 4)

    :return: List of boolean results for each file rotation
    """
    if len(files) != len(output_files):
        raise ValueError("Input and output file lists must have the same length")

    total_start_time = time.time()
    results = [None] * len(files)  # Pre-allocate list for results
    thread_times = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(rotate_svg, input_file, output_file, angle): idx
            for idx, (input_file, output_file) in enumerate(zip(files, output_files))
        }

        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            thread_id = threading.get_ident()
            thread_start_time = time.time()

            try:
                result = future.result()
                results[idx] = result
                duration = time.time() - thread_start_time
                thread_times[thread_id] = duration
                logger.debug(f"Finished rotating {files[idx]} to {output_files[idx]} in {duration:.2f} seconds")
            except Exception as exc:
                duration = time.time() - thread_start_time
                logger.warning(
                    f"Thread {thread_id} generated an exception for {files[idx]} after {duration:.2f} seconds: {exc}")
                thread_times[thread_id] = duration
                results[idx] = False

    total_duration = time.time() - total_start_time
    logger.debug(f"Parallel rotation completed in {total_duration:.2f} seconds")

    return results
