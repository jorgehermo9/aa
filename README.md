# Información de la memoria

La memoria está en `docs/memoria.pdf`

# Información del código fuente 
El código final de los boletines que se utilizó en las aproximaciones está en `src/scikit/p6.jl`, las demás carpetas en `src/` tienen el código que se fue haciendo en los distintos boletines.

# Descarga de la base de datos
Primero, descargar la base de datos [aqui](https://udcgal-my.sharepoint.com/:u:/g/personal/jorge_hermo_gonzalez_udc_es/EQZaXepDQvpIm_FtS6BLYLABbuB1WPmMrtVLPa5rLTfCbg?e=kDgxmN)

Ahora, tendremos descargado el archivo `db.zip`. Moverlo a la carpeta root de la práctica y descomprimirlo.

```console
cd aa_piano
mv /path/to/db.zip .
unzip db.zip
rm db.zip
```
## Importante
En la carpeta `dataset/` están ya los `.csv` de las características extraídas de cada aproximación que utilizamos. Al extraer las características otra vez, es posible que cambie el orden en el que los patrones están distribuídos en el `.csv` (ya que no se asume ningún orden de los archivos de la base de datos en disco), por lo que es posible que algunos resultados difieran un poco de los nuestros aunque se utilice la misma semilla aleatoria. Si se quieren obtener los mismos resultados, se recomienda utilizar los mismos datasets.

# Ejecución de las aproximaciones
Antes de nada, se va a ejecutar todo desde la carpeta root de la práctica.

## Primera aproximación
Extraer las características de la primera aproximación (que se guardarán en `dataset/aprox1.csv`) y después ejecutar el código de la primera aproximación

```console
julia src/caracteristicas/features_aprox1.jl
julia aprox1.jl
```
## Segunda aproximación

Extraer las características de la segunda aproximación (que se guardarán en `dataset/aprox2.csv`) y después ejecutar el código de la segunda aproximación

```console
julia src/caracteristicas/features_aprox2.jl
julia aprox2.jl
```

Si tarda mucho el entrenamiento de las RNAs, se puede bajar el número de ejecuciones a realizar en cada fold. Se puede cambiar con el parámetro `rna_executions` del archivo `aprox2.jl`, en la línea `56`.

## Tercera aproximación

Extraer las características de la tercera aproximación (que se guardarán en `dataset/aprox3.csv`) y después ejecutar el código de la tercera aproximación

```console
julia src/caracteristicas/features_aprox3.jl
julia aprox3.jl
```

Si tarda mucho el entrenamiento de las RNAs, se puede bajar el número de ejecuciones a realizar en cada fold. Se puede cambiar con el parámetro `rna_executions` del archivo `aprox3.jl`, en la línea `56`.

## Cuarta aproximación

Extraer las características de la cuarta aproximación (que se guardarán en `dataset/aprox4.csv`) y después ejecutar el código de la cuarta aproximación

```console
julia src/caracteristicas/features_aprox4.jl
julia aprox4.jl
```

Si tarda mucho el entrenamiento de las RNAs, se puede bajar el número de ejecuciones a realizar en cada fold. Se puede cambiar con el parámetro `rna_executions` del archivo `aprox4.jl`, en la línea `56`. En esta y en la quinta aproximación el tiempo de entrenamiento fue de más de 4h, así que recomendamos bajar el número de iteraciones aunque puedan cambiar un poco los resultados de las RNAs.

## Quinta aproximación

Para la quinta aproximación no es necesario extraer más características, se utiliza el dataset de la cuarta aproximación. Las características seleccionadas están en un array en el código de `aprox5.jl`.

```console
julia aprox5.jl
```

Si tarda mucho el entrenamiento de las RNAs, se puede bajar el número de ejecuciones a realizar en cada fold. Se puede cambiar con el parámetro `rna_executions` del archivo `aprox5.jl`, en la línea `136`.

## Sexta aproximación

Para la sexta aproximación no generaremos un `.csv`, sino un archivo de la librería jld2 que nos permita guardar la matriz de la frecuencia de cada audio. Se guardará en `dataset/db.jld2`

```console
julia src/conv/savedb.jl
julia aprox6.jl
```

Esta aproximación nos tardó aproximadamente 4h. Pero no podemos reducir el número de ejecuciones en cada fold, ya que solo hacemos 1. Ya se comentó anteriormente que puede variar el orden de lectura de los archivos, por lo que aunque se utilice la misma semilla siempre, los resultados pueden diferir un poco. En aproximaciones anteriores no era problema, pero en esta puede suponer una variación mayor, ya que sólo hacemos una ejecución de cada red en cada fold(por motivos de falta de recursos computacionales y tiempo) y la aleatoriedad puede influír mucho en los resultados. Pero esto es una suposición y esperamos que siga dando los mismos resultados.

Esta aproximación es más *verbose* que las anteriores, ya que al tardar bastante, vemos útil que se vaya mostrando como avanza el entrenamiento de cada red.