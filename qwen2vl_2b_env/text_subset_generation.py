import json

read_path_txt = "val_text_subset.txt"
read_path_json = "Annotations\\val.json"
write_path = "Annotations\\val_text.json"

with open(read_path_txt, "r", encoding="utf-8") as infile_txt, open(read_path_json, "r", encoding="utf-8") as infile_json, open(write_path, "w", encoding="utf-8") as outfile:
    # Turn image list into set for faster lookup times
    imgs = {line.strip() for line in infile_txt if line.strip()}

    # Load original json dataset
    data = json.load(infile_json)

    # JSON data is structured in list format, needs to be parsed through like a list
    subset = [block for block in data if block.get("image") in imgs]

    # Write to new file
    json.dump(subset, outfile, indent=4)





