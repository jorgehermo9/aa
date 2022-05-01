# Calcular la duración de los samples
using FFTW
using WAV
using SignalAnalysis
using Statistics


function get_duration(file::String)
	y,fs = wavread(file);

	y = y[:,1];
	
	n = length(y);
	s = n/fs;
	return s;
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

all_durations = Vector(undef,length(all_instances));

for i in 1:length(all_instances)
	(_,instance_dir) = all_instances[i]
	all_durations[i] = get_duration(instance_dir);
	println("Read instance $(instance_dir) ($(i)/$(length(all_instances))). Duration $(all_durations[i])s")
end

println("Mean duration: $(mean(all_durations)), std: $(std(all_durations))")

