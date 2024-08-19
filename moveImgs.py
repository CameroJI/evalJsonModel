import os
from os import listdir
from os.path import join, basename
import shutil
import argparse
import sys

def moveImages(archivo_txt, carpeta_destino):
    os.makedirs(carpeta_destino, exist_ok=True)

    with open(archivo_txt, 'r') as file:
        for linea in file:
            linea = linea.strip()
            
            ruta_imagen = '-'.join(linea.split('-')[:2])
            
            subcarpeta = os.path.basename(os.path.dirname(ruta_imagen))
            
            nombre_imagen = os.path.basename(ruta_imagen)
            
            nuevo_nombre = f"{subcarpeta}-{nombre_imagen}"
            
            try:
                shutil.copy(ruta_imagen, os.path.join(carpeta_destino, nuevo_nombre))
                print(f"Imagen copiada y renombrada: {nuevo_nombre}")
            except FileNotFoundError:
                print(f"Error al copiar: {ruta_imagen}")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dirPath', type=str, help='Ruta al archivo de texto con las rutas de las imágenes.', required=True)
    
    return parser.parse_args(argv)

def main(args):
    dirPath = args.dirPath
    txtList = [join(dirPath, file) for file in listdir(dirPath) if file.endswith('.txt')]
    
    for txt in txtList:
        txtName  = join(dirPath, basename(txt).split('.')[0])
        print(txtName)
        moveImages(txt, join(dirPath, txtName))

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))