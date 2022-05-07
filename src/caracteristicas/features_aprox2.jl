
using FFTW
using Statistics
using WAV
using SignalAnalysis
using DelimitedFiles


function interval(bins::Int64,max::Real)
	
	base = 2^(1/12)
	f = x -> base^x-1;

	lambda = max/f(bins);

	g = x -> lambda * f(x);

	return map(g,0:bins)
end

function freqToIndex(freq_y::AbstractArray{<:Real,1},freq::Real,fs::Real)
	n = length(freq_y);
	upperBound = fs/2;
	lowerBound = -fs/2;
	f = min(max(freq,lowerBound),upperBound);
	ind = Int(round(((f + fs/2)/fs) * n));
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
	
	freq_y = abs.(fftshift(fft(y)));
	
	# m1 = freqToIndex(freq_y,0,fs);
	# m2 = freqToIndex(freq_y,max_freq,fs);
	# target_freq = freq_y[m1:m2]
	
	E = sum((y.^2) * 1/fs);
	z_crossing = zero_crossing(y)/s;
	
	bins = 10
	bins_interval = interval(bins,max_freq);
	means = zeros(bins);
	stds = zeros(bins);
	maximums = zeros(bins);
	maximums_freq = zeros(bins);

	intervals = Vector{Tuple{Float64,Float64}}(undef,bins);
	for i in 1:bins
		lowerFreq = bins_interval[i];
		upperFreq = bins_interval[i+1];
	
		lowerInd = freqToIndex(freq_y,lowerFreq,fs);
		upperInd = freqToIndex(freq_y,upperFreq,fs);
		interval_freq = freq_y[lowerInd:upperInd]
		means[i] = mean(interval_freq);
		stds[i] =std(interval_freq);
		maximums[i] = maximum(interval_freq);
		# Acceder a la segunda ocurrencia para no coger las frecuencias negativas
		index_max_freq = findall(x->x==maximums[i],interval_freq)[1]
		maximums_freq[i] = (index_max_freq/length(interval_freq)) * (upperFreq-lowerFreq) + lowerFreq
		intervals[i] = (round(lowerFreq,digits=2),round(upperFreq,digits=2));
	end
	features = zeros(42);
	features[1] = E;
	features[2] = z_crossing;
	features[3:12] = means[:];
	features[13:22] = stds[:];
	features[23:32] = maximums[:];
	features[33:end] = maximums_freq[:];

	return features
end


# Path a la base de datos. Las clases estarán en  db_dir/<class>
db_dir = "db/piano"
classes = ["C1","C2","C3","C4","C5","C6","C7","C8","A1","A2","A3","A4","A5","A6","A7"]

# Para todas las clases:
# classes = readdir(db_dir);

all_instances = Vector{Tuple{String,String}}()
for class in classes
	class_dir =db_dir*"/"*class;
	instances = readdir(class_dir);
	for instance in instances
		feature_dir = class_dir*"/"*instance
		push!(all_instances,(class,feature_dir));
	end
end

all_features = Array{Any,2}(undef,length(all_instances),43);
for i in 1:length(all_instances)
	(instance_class,instance_dir) = all_instances[i]
	instance_features = get_features(instance_dir);
	all_features[i,1:42] = instance_features[:];
	all_features[i,43] = instance_class;	
	println("Read instance $(instance_dir) ($(i)/$(length(all_instances)))")
end

headers = ["E","zero_crossing",
"m1","m2","m3","m4","m5","m6","m7","m8","m9","m10",
"std1","std2","std3","std4","std5","std6","std7","std8","std9","std10",
"max1","max2","max3","max4","max5","max6","max7","max8","max9","max10",
"max_freq1","max_freq2","max_freq3","max_freq4","max_freq5","max_freq6","max_freq7","max_freq8","max_freq9","max_freq10",
"class"];

dataset = Array{Any,2}(undef,length(all_instances)+1,length(headers));

dataset[1,:] = headers[:];
dataset[2:end,:] = all_features[:,:];

dataset_path = "dataset/aprox2.csv"
writedlm(dataset_path, dataset, ',')

println("Dataset for classes $(classes) saved in $(dataset_path)")



