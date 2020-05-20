#!/usr/bin/python

import sys
import os
from page import Page

SHOW_STEPS = True    # change this to false if you just want to see the final output for each page.
SAVE_OUTPUT = True

inputFolder = os.path.join('../data_BCEV2/0/testpics')
outputFolder = os.path.join('../data_BCEV2/0/output_docscrum')
# /Users/murtada/Documents/BU/Arabic NLP/data_BCEV2/0/jpg_subset
print(len(os.listdir(inputFolder)))
for filename in os.listdir(inputFolder)[:]:
#for filename in ['page332.jpg', 'page335.jpg']:
    if filename == '.DS_Store':
        continue
    inputPath = os.path.join(inputFolder, filename)
    outputPath = os.path.join(outputFolder, filename)
    page = Page(inputPath, SHOW_STEPS)
    
    if SAVE_OUTPUT:
        page.save(outputPath)  # save a copy of what is displayed. Used for getting images for the paper.
    
    page.show((800, 800))

