import json
import os
from tqdm import tqdm

from mjcf2o3d.file_utils import save_json


def clean_json(json_path):
    with open(json_path, "r") as f:
        dico = json.load(f)

    cleaned_json = {}
    for partname, subdic in dico.items():
        if partname in [">FULL<", ">ORIGINAL XML<"]:
            cleaned_json[partname] = subdic
            continue

        cleaned_json[partname] = {"color": subdic["color"]}

    return cleaned_json

def batch_process(mjcf_tree):
    """
    Traverse a directory tree, find all XML files, and call the `main` function for each XML file.

    Args:
        mjcf_tree (str): The root directory of the XML file tree.
        do_visualize (bool): Whether to visualize the point cloud.
        isolate_actuators (bool): Whether to isolate actuators in the point cloud.
    """
    # Collect and filter XML file paths
    json_paths = []
    for root, _, files in os.walk(mjcf_tree):
        for file in files:
            if file.endswith(".xml"):
                xml_path = os.path.join(root, file)

                if file.startswith("mjcf2o3d"):
                    os.remove(xml_path)
                    raise Exception(f"Found invalid file. Are you sure the main faile was processed correctly? {file}")

                pcd_path = xml_path.replace(".xml", "-parsed.json")
                if os.path.exists(pcd_path):
                    json_paths.append(pcd_path)


    # Process files with tqdm progress bar
    pbar = tqdm(json_paths, desc="Processing JSON files")
    for json_file in pbar:
        pbar.set_postfix_str(json_file)
        save_json(clean_json(json_file), json_file)
        os.remove(json_file.replace("-parsed.json", "-parsed.pcd"))

if __name__ == "__main__":
    batch_process("/home/charlie/Desktop/MJCFConvert/mjcf2o3d/unimals_100")
