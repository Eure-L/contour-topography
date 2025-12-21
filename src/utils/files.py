from xml.etree import ElementTree as ET


def merge_svgs(svg_files, output_file, width=None, height=None):
    """
    Merge multiple SVG files into a single SVG.

    :param svg_files: list of SVG file paths
    :param output_file: path to save merged SVG
    :param width: width of the merged SVG (optional)
    :param height: height of the merged SVG (optional)
    """
    # Create root SVG element
    merged_svg = ET.Element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        version="1.1"
    )

    if width is not None:
        merged_svg.set("width", str(width))
    if height is not None:
        merged_svg.set("height", str(height))

    for i, svg_file in enumerate(svg_files):
        tree = ET.parse(svg_file)
        root = tree.getroot()

        # Wrap content in a group for each SVG
        g = ET.SubElement(merged_svg, "g", id=f"layer{i}")
        for child in root:
            g.append(child)

    # Save merged SVG
    tree = ET.ElementTree(merged_svg)
    tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Merged SVG saved to {output_file}")
