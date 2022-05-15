
using FFTW
using WAV
using SignalAnalysis
using JLD2, FileIO



function get_signal(file::String)
	y,fs,nbits = wavread(file);
	duration_threshold = 3
	index_threshold = Int(duration_threshold * fs);
	y = y[1:min(index_threshold,size(y,1)),:];
	
	n = length(y);
	x = 1:n;
	s = n/fs;
	# println("$(n) muestras con una frecuencia de $(fs) muestras/seg: $(n/fs) seg.")
	filename = split(file,"db/")[2]
	parent = join(split(filename,"/")[1:2],"/")
	mkpath("reduced/$(parent)")
	wavwrite(y,"reduced/$(filename)",Fs=fs,nbits=32)
	
	
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

# Utilizamos aquí como duración máxima 1.5 segundos, ya que hay algunos audios con 
# menos de dos segundos de duración. (En concreto, A#7 o B7). Esto no supone ningún problema
# en las aproximaciones anteriores, ya que los segundos de duración determinan la frecuencia
# de sampling, y, por lo tanto, el número de elementos en el array de intensidades en frecuencia/tiempo.
# En la aproximación anterior no nos hacía falta tener en cuenta que pudiese haber muestras con
# distinto sampling (si era menor de 3s, pues se utilizaba el sampling de esa duración menor), 
# Y con ese muestreo, pues se sacaban las características. Aquí es distinto, ya que necesitamos
# que todas las instancias tengan el mismo número de posiciones (n*fs), por lo que tenemos que 
# utilizar un max_duration que haga que todas las instancias tengan el mismo número de posiciones. 
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

for i in 1:length(all_instances)
	(instance_class,instance_dir) = all_instances[i]
	println("Read instance $(instance_dir) ($(i)/$(length(all_instances)))")
	get_signal(instance_dir);
end

