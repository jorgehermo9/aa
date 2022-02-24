using DelimitedFiles
using Statistics
using Flux
using Flux.Losses



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


function calculateMinMaxNormalizationParameters(data::AbstractArray{<:Real,2})
	(minimum(data,dims=1),maximum(data,dims=1))
end

function calculateZeroMeanNormalizationParameters(data::AbstractArray{<:Real,2})
	(mean(data,dims=1),std(data,dims=1))
end

## MinMmax
function normalizeMinMax!(data::AbstractArray{<:Real,2}, parameters::NTuple{2,AbstractArray{<:Real,2}})
	data_min = parameters[1];
	data_max = parameters[2];
	normalize(value,col_min,col_max) = (value-col_min)/(col_max-col_min);

	data[:,:] = normalize.(data,data_min,data_max);
end

function normalizeMinMax!(data::AbstractArray{<:Real,2})
	parameters = calculateMinMaxNormalizationParameters(data);
	normalizeMinMax!(data,parameters);
end

function normalizeMinMax(data::AbstractArray{<:Real,2},parameters::NTuple{2,AbstractArray{<:Real,2}})
	cloned = copy(data);
	return normalizeMinMax!(cloned,parameters);
end

function normalizeMinMax(data::AbstractArray{<:Real,2})
	cloned = copy(data);
	return normalizeMinMax!(cloned);
end

# Zero mean
function normalizeZeroMean!(data::AbstractArray{<:Real,2}, parameters::NTuple{2,AbstractArray{<:Real,2}})
	data_mean = parameters[1];
	data_std = parameters[2];
	normalize(value,col_mean,col_std) = (col_std==0) ? 0 : (value-col_mean)/col_std;

	data[:,:] = normalize.(data,data_mean,data_std);
end

function normalizeZeroMean!(data::AbstractArray{<:Real,2})
	parameters = calculateZeroMeanNormalizationParameters(data);
	normalizeZeroMean!(data,parameters);
end

function normalizeZeroMean(data::AbstractArray{<:Real,2},parameters::NTuple{2,AbstractArray{<:Real,2}})
	cloned = copy(data);
	normalizeZeroMean!(cloned,parameters);
	return cloned;
end

function normalizeZeroMean(data::AbstractArray{<:Real,2})
	cloned = copy(data);
	normalizeZeroMean!(cloned);
	return cloned;
end

function classifyOutputs(outputs::AbstractArray{<:Real,2},threshold::Real=0.5)
	
	rows = size(outputs,1);
	cols = size(outputs,2);
	
	if cols ==1 
		return outputs .>= threshold;
	else
		# (_,indicesMaxEachInstance) = findmax(outputs,dims=2);
		# outputs = falses(size(outputs));
		# outputs[indicesMaxEachInstance] .= true;
		#Creo que es más eficiente mi manera
		row_max = maximum(outputs,dims=2);
		return outputs.==row_max;
	end
end

function accuracy(targets::AbstractArray{Bool,1},outputs::AbstractArray{Bool,1})
	comparison = targets.==outputs;
	return mean(comparison);
end

function accuracy(targets::AbstractArray{Bool,2},outputs::AbstractArray{Bool,2})
	cols_targets = size(targets,2);
	cols_outputs = size(outputs,2);
	@assert (cols_targets==cols_outputs) "targets y outputs tienen que tener el mismo número de columnas";
	cols = cols_targets;

	@assert (cols!=2) "No puede pasar que haya dos columnas (con one-hot encoding)";

	if cols == 1
		return accuracy(targets[:,1],outputs[:,1]);
	elseif cols > 2
		classComparison = targets .==outputs;
		correctClassifications = all(classComparison,dims=2)
		return mean(correctClassifications)
	end 
end

function accuracy(targets::AbstractArray{Bool,1},outputs::AbstractArray{<:Real,1},threshold::Real=0.5)
	return accuracy(targets,outputs.>=threshold);
end

function accuracy(targets::AbstractArray{Bool,2},outputs::AbstractArray{<:Real,2})
	cols_targets = size(targets,2);
	cols_outputs = size(outputs,2);
	@assert (cols_targets==cols_outputs) "targets y outputs tienen que tener el mismo número de columnas";
	cols = cols_targets;
	
	if cols == 1
		return accuracy(targets[:,1],outputs[:,1]);
	elseif cols >2
		return accuracy(targets,classifyOutputs(outputs));
	end
end

#  unsigned int en vez de int. No tiene sentido permitir valores negativos
function createRNA(topology::AbstractArray{<:Int,1},numInputs::Int,numOutputs::Int)
	ann = Chain();
	numInputsLayer = numInputs;

	for numOutputsLayer = topology
		ann = Chain(ann...,Dense(numInputsLayer,numOutputsLayer,sigmoid));
		numInputsLayer = numOutputsLayer;
	end;

	if numOutputs == 1
		ann = Chain(ann...,Dense(numInputsLayer,numOutputs,sigmoid))

	else 
		ann = Chain(ann...,
			Dense(numInputsLayer,numOutputs,identity),
			softmax
		)
	end
	return ann;
end

function trainRNA(topology::AbstractArray{<:Int,1},dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}};
		maxEpochs::Int=1000,minLoss::Real=0,learningRate::Real=0.01)

	inputs = dataset[1];
	targets = dataset[2];
	numInputs = size(inputs,2);
	numOutputs = size(targets,2);

	ann = createRNA(topology,numInputs,numOutputs);


	loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
	loss_vector = Vector{Real}()

	for _ in 1:maxEpochs
		Flux.train!(loss,params(ann),[(inputs',targets')],ADAM(learningRate))
		current_loss = loss(inputs',targets')
		append!(loss_vector,current_loss)
		if current_loss <= minLoss
			return (ann,loss_vector)
		end
	end
	return (ann,loss_vector);
end
function trainRNA(topology::AbstractArray{<:Int,1},dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
	maxEpochs::Int=1000,minLoss::Real=0,learningRate::Real=0.01)

	inputs = dataset[1];
	targets = dataset[2];
	new_targets = reshape(targets,length(targets),1);
	new_dataset = Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}(inputs,new_targets)
	return trainRNA(topology,new_dataset,maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate);
end


dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4];
targets = dataset[:,5];

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"

inputs = convert(Array{Float32,2},inputs);
inputs = normalizeZeroMean(inputs)
targets = oneHotEncoding(targets)

(ann,loss_vector) = trainRNA([12,4],(inputs,targets),maxEpochs=1000,learningRate=0.05);

acc = accuracy(targets,ann(inputs')');

print(acc)

# Resultados
#