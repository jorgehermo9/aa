
using FFTW
using Statistics
# using Plots
using WAV
using SignalAnalysis


function interval(bins::Int64,max::Real)
	
	base = 2^(1/12)
	f = x -> base^x-1;

	lambda = max/f(bins);

	g = x -> lambda * f(x);

	return map(g,0:bins)
end

function freqToIndex(freq::Real,fs::Real)
	upperBound = fs/2;
	lowerBound = -fs/2;
	f = min(max(freq,lowerBound),upperBound);
	ind = Int(round(((f + upperBound)/fs) * n));
	return ind;
end

function zero_crossing(y::Vector{<:Real})
	crossing = 0;
	for i in 2:length(y)
		if (y[i-1]<=0 && y[i] >0) || (y[i-1]>=0 && y[i]<0)
			crossing+=1;
		end
	end
	return crossing
end

function get_features(file::String)
	y,fs = wavread(file);

	max_freq = 5000;
	
	
	duration_threshold = 3
	index_threshold = Int(duration_threshold * fs);
	y = y[:,1];
	
	y = y[1:min(index_threshold,length(y))];
	
	n = length(y);
	x = 1:n;
	s = n/fs;
	# println("$(n) muestras con una frecuencia de $(fs) muestras/seg: $(n/fs) seg.")
	
	
	
	freq_y = abs.(fftshift(fft(y)));
	
	m1 = freqToIndex(0,fs);
	m2 = freqToIndex(max_freq,fs);
	
	target_freq = freq_y[m1:m2]
	
	# plotlyjs();
	# time_graph = plot(y,label = "Time");
	f = map(x -> x * (max_freq)/length(target_freq),1:length(target_freq));
	
	# freq_graph = plot(f,target_freq, label = "Freq");
	
	
	# display(plot(time_graph,freq_graph,layout=(1,2)));
	
	
	
	E = sum((y.^2) * 1/fs);
	# mean_freq = mean(target_freq);
	# std_freq = std(target_freq);
	z_crossing = zero_crossing(y)
	
	# println("Energía de la señal: $(E)");
	# println("zero crossing/s : $(z_crossing)");
	# println("Media de la señal en frecuencia entre $(0) y $(max_freq) Hz: ", mean_freq);
	# println("Desv tipica de la señal en frecuencia entre $(0) y $(max_freq) Hz: ", std_freq);
	
	
	bins = 10
	bins_interval = interval(bins,5000);
	means = zeros(bins);
	stds = zeros(bins);
	maximums = zeros(bins);
	intervals = Vector{Tuple{Float64,Float64}}(undef,bins);
	for i in 1:bins
		lowerFreq = bins_interval[i];
		upperFreq = bins_interval[i+1];
	
		lowerInd = freqToIndex(lowerFreq,fs);
		upperInd = freqToIndex(upperFreq,fs);
		interval_freq = freq_y[lowerInd:upperInd]
		means[i] = mean(interval_freq);
		stds[i] =std(interval_freq);
		maximums[i] = maximum(interval_freq);
		intervals[i] = (round(lowerFreq,digits=2),round(upperFreq,digits=2));
	end
	# println(round.(means,digits=2));
	# println(round.(stds,digits=2));
	# println(round.(maximums,digits=2));
	# println(intervals);

	features = zeros(1+1+30);
	features[1] = E;
	features[2] = z_crossing;
	features[3:12] = means[:];
	features[13:22] = stds[:];
	features[23:end] = maximums[:];

	return features;
end

# Creamos una señal de n muestras: es un array de flotantes
# y,fs = wavread("/home/jorge/github/aa/db/a4/Piano-A4-25.wav");

db_dir = "/home/jorge/github/aa/db"
classes = readdir(db_dir);

for class in classes
	class_dir =db_dir*"/"*class;
	instances = readdir(class_dir);
	for instance in instances
		feature_dir = class_dir*"/"*instance
		features = get_features(feature_dir);
		println(feature_dir)
	end
end







