"""
Script to transform all xml files in a certain directory to txt.
"""

import os
from lxml import etree

ENCODING = "utf-8"
DIRECTORY = "data/Bundestagsprotokolle"

for file in os.listdir(DIRECTORY):
    if file.endswith(".xml"):
        print("converting:", file)
        name = file.rsplit(".", 1)[0]
        tree = etree.parse(os.path.join(DIRECTORY, file))
        notags = etree.tostring(tree, encoding=ENCODING, method='text')
        with open(os.path.join(DIRECTORY, name) + ".txt", "w", encoding=ENCODING) as text_file:
            text_file.write(str(notags, ENCODING))

