
using FFTW
using Statistics
using Plots
using WAV

# Que frecuenicas queremos coger
f1 = 700; f2 = 3000;

# Creamos una señal de n muestras: es un array de flotantes
y,fs = wavread("/home/jorge/github/aa/db/a4/Piano-A4-25.wav");

y = y[:,1]
n = length(y);
x = 1:n;
println("$(n) muestras con una frecuencia de $(fs) muestras/seg: $(n/fs) seg.")




# Hallamos la FFT y tomamos el valor absoluto
freq_y = abs.(fftshift(fft(y)));
freq_y = freq_y./maximum(freq_y);

# Los valores absolutos de la primera mitad de la señal deberian de ser iguales a los de la segunda mitad, salvo errores de redondeo
# Esto se puede ver en la grafica:

# Representamos la señal
plotlyjs();
time_graph = plot(y,label = "Time");
f = map(x -> x * fs/n,1:length(freq_y))
f = f.-fs/2;
freq_graph = plot(f,freq_y, label = "Freq");
display(plot(time_graph,freq_graph,layout=(1,2)));

# #  pero ademas lo comprobamos en el codigo
# if (iseven(n))
#     @assert(mean(abs.(senalFrecuencia[2:Int(n/2)] .- senalFrecuencia[end:-1:(Int(n/2)+2)]))<1e-8);
#     senalFrecuencia = senalFrecuencia[1:(Int(n/2)+1)];
# else
#     @assert(mean(abs.(senalFrecuencia[2:Int((n+1)/2)] .- senalFrecuencia[end:-1:(Int((n-1)/2)+2)]))<1e-8);
#     senalFrecuencia = senalFrecuencia[1:(Int((n+1)/2))];
# end;

# # Grafica con la primera mitad de la frecuencia:
# graficaFrecuenciaMitad = plot(senalFrecuencia, label = "");

# # Representamos las 3 graficas juntas
# display(plot(graficaTiempo, graficaFrecuencia, graficaFrecuenciaMitad, layout = (3,1)));


# # A que muestras se corresponden las frecuencias indicadas
# #  Como limite se puede tomar la mitad de la frecuencia de muestreo
# m1 = Int(round(f1*2*length(senalFrecuencia)/fs));
# m2 = Int(round(f2*2*length(senalFrecuencia)/fs));

# # Unas caracteristicas en esa banda de frecuencias
# println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(senalFrecuencia[m1:m2]));
# println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(senalFrecuencia[m1:m2]));


#keeps plot open
read(STDIN,Char);