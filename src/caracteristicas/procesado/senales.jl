
using FFTW
using Statistics

# Frecuencia de muestreo, por ejemplo: 100 Hz
Fs = 100;
# Numero de muestras
n = 200;
# Que frecuenicas queremos coger
f1 = 10; f2 = 20;

println("$(n) muestras con una frecuencia de $(Fs) muestras/seg: $(n/Fs) seg.")

# Creamos una señal de n muestras: es un array de flotantes
x = 1:n;
senalTiempo = sin.(x)./x;

# Representamos la señal
using Plots; plotlyjs();
graficaTiempo = plot(x, senalTiempo, label = "", xaxis = x);

# Hallamos la FFT y tomamos el valor absoluto
senalFrecuencia = abs.(fft(senalTiempo));



# Los valores absolutos de la primera mitad de la señal deberian de ser iguales a los de la segunda mitad, salvo errores de redondeo
# Esto se puede ver en la grafica:
graficaFrecuencia = plot(senalFrecuencia, label = "");
#  pero ademas lo comprobamos en el codigo
if (iseven(n))
    @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
    senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
else
    @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
    senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
end;

# Grafica con la primera mitad de la frecuencia:
graficaFrecuenciaMitad = plot(senalFrecuencia, label = "");


# Representamos las 3 graficas juntas
display(plot(graficaTiempo, graficaFrecuencia, graficaFrecuenciaMitad, layout = (3,1)));


# A que muestras se corresponden las frecuencias indicadas
#  Como limite se puede tomar la mitad de la frecuencia de muestreo
m1 = Int(round(f1*2*length(senalFrecuencia)/Fs));
m2 = Int(round(f2*2*length(senalFrecuencia)/Fs));

# Unas caracteristicas en esa banda de frecuencias
println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(senalFrecuencia[m1:m2]));
println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(senalFrecuencia[m1:m2]));
