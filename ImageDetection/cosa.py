import json
from PIL import Image

with open("/home/juanjo/Desktop/TFM/export-2020-02-04T14_06_32.168Z.json", "r") as f:
    json_object = json.load(f)
    
    
with open("/home/juanjo/Desktop/TFM/labels.data", "w+") as f:
    for img in json_object:
        img_name = img["External ID"]
        labels = img["Label"]
        
        if "objects" in labels:
            for l in labels["objects"]:
                value = l["value"]
                bbox = l["bbox"]
                
                f.write("/media/juanjo/ADATA SD600Q/test/{},{},{},{},{},{}\n".format(img_name, bbox["left"], bbox["top"], bbox["left"] + bbox["width"], bbox["top"] + bbox["height"], value))
        else:
            im = Image.open("/media/juanjo/ADATA SD600Q/test/{}".format(img_name))
            width, height = im.size
            f.write("/media/juanjo/ADATA SD600Q/test/{},{},{},{},{},bg\n".format(img_name, 0, 0, width, height))
