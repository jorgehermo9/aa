
using FFTW
using Statistics
using Plots
using WAV

# Que frecuenicas queremos coger
f1 = 500; f2 = 2000;

# Creamos una señal de n muestras: es un array de flotantes
# y,fs = wavread("/home/jorge/github/aa/db/a4/Piano-A4-25.wav");
y,fs = wavread("/home/jorge/github/aa/db/c4/Piano-C4-21.wav");

# y = yo[:,1]
# Coger primer x primeros segundos

duration_threshold = 2
y = y[1:Int(duration_threshold * fs),1]

n = length(y);
x = 1:n;
s = n/fs;
println("$(n) muestras con una frecuencia de $(fs) muestras/seg: $(n/fs) seg.")



freq_y = abs.(fftshift(fft(y)));
f1 = min(max(f1,-fs/2),fs/2);
f2 = min(max(f2,-fs/2),fs/2);
m1 = Int(round(((f1+fs/2)/fs)*n));
m2 = Int(round(((f2+fs/2)/fs)*n));

index_y = map(x -> x>m1 && x<m2,1:length(freq_y));

freq_y = freq_y[index_y];
plotlyjs();
time_graph = plot(y,label = "Time");
f = map(x -> x * fs/n,1:length(freq_y));
f = f.-fs/2;
freq_graph = plot(f,freq_y, label = "Freq");
display(plot(time_graph,freq_graph,layout=(1,2)));


println("Media de la señal en frecuencia entre $(f1) y $(f2) Hz: ", mean(freq_y[m1:m2]));
println("Desv tipica de la señal en frecuencia entre $(f1) y $(f2) Hz: ", std(freq_y[m1:m2]));

E = sum((y.^2) * 1/fs);
function zero_crossing(y::Vector{<:Real})
	crossing = 0;
	for i in 2:length(y)
		if (y[i-1]<=0 && y[i] >0) || (y[i-1]>=0 && y[i]<0)
			crossing+=1;
		end
	end
	return crossing
end
println("zero crossing/s : $(zero_crossing(y)/s)");
println("Energía de la señal: $(E)");

println("Desv tipica de la señal en tiempo: ", std(y));


