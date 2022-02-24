using DelimitedFiles
using Statistics

dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4];
targets = dataset[:,5];

inputs = convert(Array{Float32,2},inputs);
classes = unique(targets);


if length(classes)>2
	onehot = Array{Bool,2}(undef,length(targets),length(classes))

	for i = 1:length(classes)
		onehot[:,i] = classes[i].==targets;
	end
else
	onehot = classes[1].==targets;
end

# inputs_min = minimum(inputs,dims=1);
# inputs_max = maximum(inputs,dims=1);
# inputs_mean = mean(inputs,dims=1);
# inputs_std = std(inputs,dims=1);



function normalize(elem,mean,std)
	if std == 0
		return 0
	else
		return (elem-mean)/std;
	end
	
end

function denormalize(elem,mean,std)
	elem*std + mean
end

input_mean = mean(inputs,dims=1);
input_std = std(inputs,dims=1);

new_inputs = Array{Float32,2}(undef,size(inputs))
for i in 1:size(inputs,2)
	new_inputs[:,i] = normalize.(inputs[:,i],input_mean[i],input_std[i])
end

denormalized_inputs = Array{Float32,2}(undef,size(inputs))
for i in 1:size(inputs,2)
	denormalized_inputs[:,i] = denormalize.(new_inputs[:,i],input_mean[i],input_std[i])
end
print(denormalized_inputs.-inputs)
# onehot(class,value) = class.==value
# targets = onehot.((classes,),targets)

# reduce(hcat,targets)'

#targets = [f ==e for e in targets,f in classes];
#print(targets)
