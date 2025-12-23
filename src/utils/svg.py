import os.path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from xml.etree import ElementTree as ET


def convert_stroke_to_path(stroke_element: ET.Element):
    """
    Convert an SVG stroke element to a path element.

    Args:
        stroke_element: An ElementTree Element representing a stroke (line, polyline, etc.)

    Returns:
        A new ElementTree Element representing the equivalent path
    """
    # Create new path element
    path = ET.Element("path")

    # Copy common attributes
    for attr in ['id', 'class', 'style']:
        if attr in stroke_element.attrib:
            path.attrib[attr] = stroke_element.attrib[attr]

    # Handle different stroke types
    tag = stroke_element.tag
    if tag.endswith('line'):
        # Convert line to path
        x1 = float(stroke_element.attrib['x1'])
        y1 = float(stroke_element.attrib['y1'])
        x2 = float(stroke_element.attrib['x2'])
        y2 = float(stroke_element.attrib['y2'])
        d = f"M {x1},{y1} L {x2},{y2}"
        path.attrib['d'] = d

    elif tag.endswith('polyline') or tag.endswith('polygon'):
        # Convert polyline/polygon to path
        points = stroke_element.attrib['points'].strip().split()
        path_parts = []
        for i, point in enumerate(points):
            x, y = point.split(',')
            if i == 0:
                path_parts.append(f"M {x},{y}")
            else:
                path_parts.append(f"L {x},{y}")
        if tag.endswith('polygon'):
            path_parts.append("Z")  # Close the path
        path.attrib['d'] = " ".join(path_parts)

    # Copy stroke styling attributes
    for attr in ['stroke', 'stroke-width', 'stroke-linecap', 'stroke-linejoin',
                 'stroke-dasharray', 'stroke-opacity', 'fill', 'fill-opacity']:
        if attr in stroke_element.attrib:
            path.attrib[attr] = stroke_element.attrib[attr]

    return path


def convert_strokes_to_paths_in_svg(svg_file: str, select_attr: str = 'all', max_workers: int = 4) -> bool:
    """
    Convert SVG strokes to paths directly in Python.

    Args:
        svg_file: Path to the input SVG file
        select_attr: CSS selector for elements to convert (default: 'all')

    Returns:
        True if conversion was successful, False otherwise
    """
    try:
        # Parse the SVG file
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Find elements to convert
        if select_attr == 'all':
            elements = root.findall(".//*")
        else:
            elements = root.findall(f".//*{select_attr}")

        # Use ThreadPoolExecutor to process elements in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_elem = {
                executor.submit(convert_stroke_to_path, elem): elem
                for elem in elements
            }

            # Collect results as they complete
            for future in as_completed(future_to_elem):
                elem = future_to_elem[future]
                try:
                    path = future.result()
                    root.replace(elem, path)
                    print(f"Thread {threading.get_ident()} finished processing element in {svg_file}")
                except Exception as exc:
                    print(f"Thread {threading.get_ident()} generated an exception for element in {svg_file}: {exc}")
                    return False

        # Write the modified SVG back to the file
        svg_dir, svg_name = os.path.split(svg_file)
        new_file = os.path.join(svg_dir,f"trace_{svg_name}")

        tree.write(new_file, encoding="utf-8", xml_declaration=True)
        return True

    except Exception as e:
        print(f"Error converting SVG: {e}", file=sys.stderr)
        return False


def parallel_convert_strokes_to_paths_in_svg(files: List[str], select_attr: str = 'all', max_workers: int = 4) -> List[
    bool]:
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
            executor.submit(convert_strokes_to_paths_in_svg, file, select_attr, max_workers): file
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
