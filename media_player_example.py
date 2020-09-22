import PySimpleGUI as sg
import os

cwd = os.getcwd()
fname = 'download.png'

with open('{}/{}'.format(cwd, fname)) as fh:
    image1 = fh.read()

[sg.Image(data=image1, key='key1', size=(5, 6))]