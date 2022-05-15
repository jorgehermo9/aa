
using FFTW
using Statistics
using Plots
using WAV


function freqToIndex(freq_y::AbstractArray{<:Real,1},freq::Real,fs::Real)
	n = length(freq_y);
	upperBound = fs/2;
	lowerBound = -fs/2;
	f = min(max(freq,lowerBound),upperBound);
	ind = Int(round(((f + upperBound)/fs) * n)+1);
	return ind;
end

nota="F#1"
id="98"
archivo="Piano-$(nota)-$(id).wav"
y,fs = wavread("db/piano/$(nota)/$(archivo)");

max_freq = 5000;
duration_threshold = 3
index_threshold = Int(duration_threshold * fs);
y = y[:,1];

y = y[1:min(index_threshold,length(y))];

n = length(y);
x = 1:n;
s = n/fs;
println("$(n) muestras con una frecuencia de $(fs) muestras/seg: $(n/fs) seg.")



freq_y = abs.(fftshift(fft(y)));

m1 = freqToIndex(freq_y,0,fs);
m2 = freqToIndex(freq_y,max_freq,fs);

target_freq = freq_y[m1:m2]

t_ticks = string.(1:Int(s));
plotlyjs();
time_graph = plot(y,
	label = "Time",
	xaxis="t[s]",
	xticks=0:fs:fs*s,
	xformatter = ((x) -> "$(round(Int,x/fs))"));
f = map(x -> x * (max_freq)/length(target_freq),1:length(target_freq));

freq_graph = plot(f,target_freq, label = "Freq",xaxis="f[hz]",c=:orange);

graph = plot(time_graph,freq_graph,layout=(1,2));
#display(graph);
# savefig(time_graph,"$(nota)_time_graph.svg")
# savefig(freq_graph,"$(nota)_freq_graph.svg")
savefig(graph,"$(nota)_graph.svg")
println("Saved graph to $(nota)_graph.svg")




