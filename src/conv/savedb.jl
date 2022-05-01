
using FFTW
using WAV
using SignalAnalysis
using JLD2, FileIO


function oneHotEncoding(feature::AbstractArray{<:Any,1}, classes::AbstractArray{<:Any,1})
	if length(classes)>2
		onehot = Array{Bool,2}(undef,length(feature),length(classes))
	
		for i = 1:length(classes)
			onehot[:,i] = classes[i].==feature;
		end
		return onehot
	else
		onehot = classes[1].==feature;
		return reshape(onehot,(length(onehot),1))
	end
end

oneHotEncoding(feature::AbstractArray{<:Any,1}) = oneHotEncoding(feature,unique(feature))
oneHotEncoding(feature::AbstractArray{Bool,1}) =reshape(feature,(length(feature),1))



function freqToIndex(freq_y::AbstractArray{<:Real,1},freq::Real,fs::Real)
	n = length(freq_y);
	upperBound = fs/2;
	lowerBound = -fs/2;
	f = min(max(freq,lowerBound),upperBound);
	ind = Int(round(((f + fs/2)/fs) * n));
	return ind;
end

function get_signal(file::String)
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
	
	m1 = freqToIndex(freq_y,0,fs);
	m2 = freqToIndex(freq_y,max_freq,fs);
	
	target_freq = freq_y[m1:m2]
	# display(target_freq)
	
	# plotlyjs();
	# time_graph = plot(y,label = "Time");
	# f = map(x -> x * (max_freq)/length(target_freq),1:length(target_freq));
	
	# freq_graph = plot(f,target_freq, label = "Freq");
	
	
	# display(plot(time_graph,freq_graph,layout=(1,2)));
	return target_freq;
end


db_dir = "/home/jorge/github/aa/db/piano"
classes = readdir(db_dir);

all_instances = Vector{Tuple{String,String}}()
for class in classes
	class_dir =db_dir*"/"*class;
	instances = readdir(class_dir);
	for instance in instances
		feature_dir = class_dir*"/"*instance
		push!(all_instances,(class,feature_dir));
	end
end

fs  = 48000
max_duration = 3
n = fs*max_duration
max_freq = 5000
m1 = (0+fs/2)/fs * n
m2 = (5000+fs/2)/fs * n
n_target_freq = Int(round(m2-m1+1))

classes_unique = unique(classes)
display(classes_unique)

all_signals = Array{Any,2}(undef,n_target_freq,length(all_instances));
all_labels = Vector(undef,length(all_instances));

for i in 1:length(all_instances)
	(instance_class,instance_dir) = all_instances[i]
	all_labels[i] = instance_class
	all_signals[:,i] = get_signal(instance_dir);
	println("Read instance $(instance_dir) ($(i)/$(length(all_instances)))")
end

all_labels_onehot = oneHotEncoding(all_labels,classes_unique)'

# display(all_signals)
# display(all_labels_onehot)


path = "db.jld2"
@save path all_signals all_labels all_labels_onehot

println("Saved signal and labels to $(path)")

