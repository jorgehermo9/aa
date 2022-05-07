
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
	
	duration_threshold = 0.1
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
	
	target_freq = freq_y[m1:(m2-1)]
	return target_freq;
end


db_dir = "db/piano"

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

# Decidimos utilizar como duración máxima 0.1 segundos. Con más muestreo en frecuencia,
# tardaba muchísimo. Antes utilizabamos un array de 7500 posiciones, ahora 500. 
# Pensamos que va a funcionar bien también así. Si funciona bien con 0.1 segundos, aún mejor.
fs  = 48000
max_duration = 0.1
n = fs*max_duration
max_freq = 5000
m1 = (0+fs/2)/fs * n
m2 = (5000+fs/2)/fs * n
n_target_freq = Int(round(m2-m1))

classes_unique = unique(classes)
display(classes_unique)

println("Vector size: $(n_target_freq)")
all_signals = Array{Float32,2}(undef,n_target_freq,length(all_instances));
all_labels = Vector(undef,length(all_instances));
for i in 1:length(all_instances)
	(instance_class,instance_dir) = all_instances[i]
	all_labels[i] = instance_class
	println("Read instance $(instance_dir) ($(i)/$(length(all_instances)))")
	all_signals[:,i] = get_signal(instance_dir);
end

all_labels_onehot = oneHotEncoding(all_labels,classes_unique)'

path = "dataset/db.jld2"
@save path all_signals all_labels all_labels_onehot

println("Saved signal and labels to $(path)")

