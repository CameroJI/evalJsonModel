from os import listdir, makedirs
from os.path import join, basename, dirname
from shutil import copy, move
import argparse
import sys

def moveImages(archivo_txt, carpeta_destino):
    makedirs(carpeta_destino, exist_ok=True)

    with open(archivo_txt, 'r') as file:
        for linea in file:
            linea = linea.strip()
            
            ruta_imagen = linea.split('-')[0]
            
            subcarpeta = basename(dirname(ruta_imagen))
            
            nombre_imagen = basename(ruta_imagen)
            
            nuevo_nombre = f"{subcarpeta}-{nombre_imagen}"
            
            try:
                copy(ruta_imagen, join(carpeta_destino, nuevo_nombre))
                print(f"Imagen copiada y renombrada: {nuevo_nombre}")
            except FileNotFoundError:
                print(f"Error al copiar: {ruta_imagen}")
    
    txtFileNew = join(dirname(dirname(archivo_txt)), basename(archivo_txt))
    move(archivo_txt, txtFileNew)
    
                
def txtGetFiles(path):
    txtList = [join(path, file) for file in listdir(path) if file.endswith('.txt')]
    
    for txt in txtList:
        txtName  = basename(txt).split('.')[0]
        print(txtName)
        moveImages(txt, join(dirname(path), txtName))

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dirPath', type=str, help='Ruta al archivo de texto con las rutas de las imágenes.', required=True)
    
    return parser.parse_args(argv)

def main(args):
    dirPath = args.dirPath
    
    txtGetFiles(dirPath)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))