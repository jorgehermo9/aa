
using FileIO
using Images

# Documentacion en:
#   https://juliaimages.org/
#   https://juliaimages.org/latest/function_reference/

# Cargar una imagen
imagen = load("lena_std.tiff");
imagen = load("lena.jpg");

# Mostrar esa imagen, de cualquiera de estas dos formas
display(imagen)
# using ImageView; imshow(imagen);

# Que tipo tiene la imagen (es un array)
typeof(imagen)
# Array{RGB{Normed{UInt8,8}},2}:
#   Array bidimensional: Array{    ,2}
#   donde cada elemento es de tipo RGB{Normed{UInt8,8}}
# Tamaño de la imagen: el tamaño del array
size(imagen)
# Por ejemplo, para ver el primer pixel:
dump(imagen[1,1])
# Tipo: RGB{Normed{UInt8,8}}
typeof(imagen[1,1])
# Cada pixel tiene 3 campos: r,g,b
# Cada campo es del tipo indicado
#  Normed{UInt8,8}): 8 bits, normalizado entre 0 y 1
#  Por ejemplo, para comar la componente roja, de cualquiera de estas formas
imagen[1,1].r
red(imagen[1,1])
# Las otras dos componentes, de igual manera:
imagen[1,1].g
green(imagen[1,1])
imagen[1,1].b
blue(imagen[1,1])
# Para crear un elemento de tipo RGB, simplemente instanciar RGB indicando los 3 componentes, por ejemplo, el blanco:
RGB(1,1,1)


# Para extraer un canal de la imagen, hacer un broadcast de la operacion correspondiente que se realiza a un pixel, pero a toda la matriz
#  red(pixeles) -> devuelve la componente roja de ese pixel
#  red.(array de pixeles) -> devuelve un array del mismo tamaño, con las componente rojas de esos pixeles
# Por ejemplo:
matrizRojos = red.(imagen);
# Para construir una imagen solamente con ese canal, hacer una operacion de broadcast
#  RGB(         1,        0, 0 ) -> devuelve el color rojo (solo un pixel)
#  RGB.( [0.1, 0.5, 0.9], 0, 0 ) -> devuelve un array de 3 elementos (es decir, una imagen de 1 fila y 3 columnas) con esos colores. Esta linea es equivalente a:
#  RGB.( [0.1, 0.5, 0.9], [0, 0, 0], [0, 0, 0] )
# Por tanto, para construir la imagen solo con el canal rojo
imagenRojo = RGB.(matrizRojos,0,0)
# De esta forma, la imagen original se pueden extraer sus 3 canales (rojo, verde y azul) y recomponerla de la siguiente manera:
RGB.(red.(imagen), green.(imagen), blue.(imagen))



# Convertir la imagen a escala de grises:
#  Gray(pixel) -> convierte un color RGB a escala de grises
# Convertir toda la imagen: hacer un broadcast
imagenBN = Gray.(imagen);

# Tipo de esta imagen en escala de grises:
typeof(imagenBN)
#  Array{Gray{Normed{UInt8,8}},2}
# Para ver el primer pixel:
dump(imagenBN[1,1])
# Para tomar su valor:
imagenBN[1,1].val
gray(imagenBN[1,1])
# Para tomar todos los valores: se hace un broadcast de esta operacion
matrizBN = Gray.(imagenBN);


######################################################################################################################
# Caracteristicas morfologicas de imagenes o partes de imagenes:
# Cargamos la imagen
imagen = load("calle.jpg"); display(imagen);

# Vamos a detectar los objetos rojos
#  Aquellos cuyo valor de rojo es superior en cierta cantidad al valor de verde y azul
# Definimos en que cantidad queremos que sea mayor
diferenciaRojoVerde = 0.3; diferenciaRojoAzul = 0.3;
canalRojo = red.(imagen); canalVerde = green.(imagen); canalAzul = blue.(imagen);
matrizBooleana = (canalRojo.>(canalVerde.+diferenciaRojoVerde)) .& (canalRojo.>(canalAzul.+diferenciaRojoAzul));
# Mostramos esta matriz booleana para ver que objetos ha encontrado
display(Gray.(matrizBooleana));

# Esto se podria haber hecho, de forma similar, con el siguiente codigo, definiendo primero la funcion a aplicar en todos los pixeles:
esPixelRojo(pixel::RGB) = (pixel.r > pixel.g + diferenciaRojoVerde) && (pixel.r > pixel.b + diferenciaRojoAzul);
# Y despues aplicando esa funcion a toda la imagen haciendo un broadcast:
matrizBooleana = esPixelRojo.(imagen);
display(Gray.(matrizBooleana));

# La siguiente funcion transforma un array booleano (imagen umbralizada) en un array de etiquetas
# Cada grupo de píxeles puesto como "true" en la matriz booleana y conextados se le asigna una etiqueta
# Por ejemplo, la imagen umbralizada
# 0 0 0 0
# 0 1 1 0
# 0 0 1 0
# 0 0 0 0
# 0 1 0 0
# 0 1 0 0
#  contiene 2 objetos, cada pixel se etiqueta como objeto "1", "2", o "0" (ninguno)
labelArray = ImageMorphology.label_components([ 0 0 0 0;
                                                0 1 1 0;
                                                0 0 1 0;
                                                0 0 0 0;
                                                0 1 0 0;
                                                0 1 0 0])
# Resultado:
# 0  0  0  0
# 0  1  1  0
# 0  0  1  0
# 0  0  0  0
# 0  2  0  0
# 0  2  0  0

# Aplicamos esta funcion a la matriz booleana (imagen umbralizada) que construimos antes:
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos")

# A partir de aqui se pueden extraer distintas caracteristicas, como pueden las siguientes:
#  Devuelven una caracteristica por cada etiqueta distinta, incluyendo como primera la etiqueta "0"
boundingBoxes = ImageMorphology.component_boxes(labelArray);
tamanos = ImageMorphology.component_lengths(labelArray);
pixeles = ImageMorphology.component_indices(labelArray);
pixeles = ImageMorphology.component_subscripts(labelArray);
centroides = ImageMorphology.component_centroids(labelArray);

# Sin embargo, suele ser util filtrar los objetos en primer lugar y eliminar los muy grandes o muy pequeños
# Calculamos los tamaños
tamanos = component_lengths(labelArray);
# Que etiquetas son de objetos demasiado pequeños (30 pixeles o menos):
etiquetasEliminar = findall(tamanos .<= 30) .- 1; # Importate el -1, porque la primera etiqueta es la 0
# Se construye otra vez la matriz booleana, a partir de la matriz de etiquetas, pero eliminando las etiquetas indicadas
# Para hacer esto, se hace un bucle sencillo en el que se itera por cada etiqueta
#  Esto se realiza de forma sencilla con la siguiente linea
matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
display(Gray.(matrizBooleana));


# Con esos objetos rojos "grandes", se toman de nuevo las etiquetas
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos rojos grandes")

# Vamos a situar el centroide de estos objetos en la imagen umbralizada, poniéndolo en color rojo
# Por tanto, hay que construir una imagen en color:
imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
# Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
centroides = ImageMorphology.component_centroids(labelArray)[2:end];
# Para cada centroide, ponemos su situacion en color rojo
for centroide in centroides
    x = Int(round(centroide[1]));
    y = Int(round(centroide[2]));
    imagenObjetos[ x, y ] = RGB(1,0,0);
end;

# Vamos a recuadrar el bounding box de estos objetos, en color verde
# Calculamos los bounding boxes, y eliminamos el primero (el objeto "0")
boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
display(imagenObjetos);

# Y hacemos lo mismo con la imagen original:
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagen[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagen[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagen[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagen[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
display(imagen);


# Finalmente, guardamos esta imagen
# save("imagenProcesada.jpg", imagen)
