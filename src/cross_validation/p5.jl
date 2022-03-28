using DelimitedFiles
using Statistics
using Flux
using Flux.Losses
using Random
using Plots

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
		validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Matrix{Real}(undef,0,0),Matrix{Bool}(undef,0,0)),
		test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}}=(Matrix{Real}(undef,0,0),Matrix{Bool}(undef,0,0)),
		maxEpochsVal::Int=20,maxEpochs::Int=1000,minLoss::Real=0,learningRate::Real=0.01)

	inputs = dataset[1];
	targets = dataset[2];
	numInputs = size(inputs,2);
	numOutputs = size(targets,2);

	(validation_inputs,validation_targets) = validation;
	has_validation = length(validation_inputs) >0
	
	(test_inputs,test_targets) = test;
	has_test = length(test_inputs) >0


	ann = createRNA(topology,numInputs,numOutputs);

	loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
	train_loss_vector = Vector{Real}();
	val_loss_vector = Vector{Real}();
	test_loss_vector = Vector{Real}();


	
	push!(train_loss_vector,loss(inputs',targets'));
	if has_validation
		best_loss_val = loss(validation_inputs',validation_targets');
		push!(val_loss_vector,best_loss_val);
		best_rna_val = deepcopy(ann);
	end
	if has_test
		push!(test_loss_vector,loss(test_inputs',test_targets'));
	end
	
	
	
	epochs_val = 0;

	for _ in 1:maxEpochs
		Flux.train!(loss,params(ann),[(inputs',targets')], ADAM(learningRate))
		train_loss = loss(inputs',targets')
		push!(train_loss_vector,train_loss)

		if has_validation
			validation_loss = loss(validation_inputs',validation_targets');
			if validation_loss < best_loss_val
				best_loss_val = validation_loss;
				best_rna_val = deepcopy(ann);
				epochs_val=0;
			else
				epochs_val=epochs_val+1;
			end
			push!(val_loss_vector,validation_loss);
		end
		
		if has_test
			push!(test_loss_vector,loss(test_inputs',test_targets'));
		end
		

		if train_loss <= minLoss || epochs_val >= maxEpochsVal
			break;
		end
	end

	if has_validation
		returned_rna = best_rna_val;
	else 
		returned_rna = ann;
	end

	return (returned_rna,train_loss_vector,val_loss_vector,test_loss_vector)
end
function trainRNA(topology::AbstractArray{<:Int,1},dataset::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}};
	validation::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Matrix{Real}(undef,0,0),Vector{Bool}()),
	test::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,1}}=(Matrix{Real}(undef,0,0),Vector{Bool}()),
	maxEpochsVal::Int=20,maxEpochs::Int=1000,minLoss::Real=0,learningRate::Real=0.01)

	(inputs,targets) = dataset;
	(validation_inputs,validation_targets) = validation;
	(test_inputs,test_targets) = test;

 	new_targets = reshape(targets,length(targets),1);
	new_val_targets = reshape(validation_targets,length(validation_targets),1);
	new_test_targets = reshape(test_targets,length(test_targets),1);

	new_dataset = (inputs,new_targets);
	new_validation = (validation_inputs,new_val_targets);
	new_test = (test_inputs,new_test_targets);

	return trainRNA(topology,new_dataset,maxEpochs=maxEpochs,minLoss=minLoss,learningRate=learningRate,
		validation=new_validation,test=new_test,maxEpochsVal=maxEpochsVal);
end


###### P3

function holdOut(N::Int,P::Real)
	@assert P>=0 && P<=1;
	num_elems = convert(Int,round(N * P));	
	index = randperm(N);
	return (index[num_elems+1:end], index[1:num_elems]);
end

function holdOut(N::Int,Pval::Real,Ptest::Real)
	@assert Pval>=0 && Ptest>=0 && Pval+Ptest<=1;

	Ptrain = 1 - (Pval + Ptest);

	(no_train,train) = holdOut(N,Ptrain);
	M = length(no_train);

	Ptest_new = Ptest/(1-Ptrain);

	(val_aux,test_aux) = holdOut(M,Ptest_new);

	val = no_train[val_aux];
	test = no_train[test_aux];

	@assert length(train)+length(val)+length(test)==N;
	return (train, val, test);
end

###### P4

function confusionMatrix(outputs::AbstractArray{Bool,1},targets::AbstractArray{Bool,1})
	
	matrix_values(out,targ) = targ*2+out

	# Transform pairs of values to flat index on the confusion matrix (0,1,2,3)
	values = matrix_values.(outputs,targets);

	confusion_matrix = Matrix{Int}(undef,2,2);
	confusion_matrix[1,1] = sum(values.==0);
	confusion_matrix[1,2] = sum(values.==1);
	confusion_matrix[2,1] = sum(values.==2);
	confusion_matrix[2,2] = sum(values.==3);

	VN = confusion_matrix[1,1];
	FP = confusion_matrix[1,2];
	FN = confusion_matrix[2,1];
	VP = confusion_matrix[2,2];

	if VN == length(targets)
		recall = 1;
		VPP = 1;
	else
		if (VP+FN) == 0 #There is no positive targets
			recall = 0;
		else
			recall = VP/(VP+FN);
		end

		if (VP+FP) == 0 # If there is no positive output
			VPP = 0;
		else
			VPP = VP/(VP+FP);
		end
	end

	if VP == length(targets)
		specifity = 1;
		VPN = 1;
	else
		if (VN+FP) == 0 #There is no negative targets
			specifity =0;
		else
			specifity = VN/(VN+FP);
		end
		if (VN + FN) == 0 #There is no negative outputs
			VPN = 0;
		else
			VPN = VN/(VN+FN);
		end
	end

	
	
	if (VN + VP+FP+VN) ==0 #There is no target. length(target) == 0
		acc = 0;
	else
		acc = (VN+VP)/(VN+VP+FP+FN);
	end

	err = 1 - acc;
	if (recall + VPP) == 0 # If VP ==0
		F1=0;
	else
		F1 = 2 * (recall*VPP/(recall+VPP));
	end

	return (acc,err,recall,specifity,VPP,VPN,F1,confusion_matrix);
end

function confusionMatrix(outputs::AbstractArray{<:Real,1},targets::AbstractArray{Bool,1};threshold::Real=0.5)
	classified_outputs = outputs.>= threshold;
	return confusion_matrix(classified_outputs,targets);
end

@enum Strat begin
	macro_strat
	weighted_strat
end
function confusionMatrix(outputs::AbstractArray{Bool,2},targets::AbstractArray{Bool,2},strat::Strat)

	@assert size(outputs,2) == size(targets,2) && size(targets,2) != 2 

	numInstances = size(targets,1)
	numClasses = size(targets,2)
	
	if numClasses ==1
		vector_outputs = outputs[:,1];
		vector_targets = targets[:,1];
		return confusionMatrix(vector_outputs,vector_targets);
	end
	recall_vec = zeros(numClasses);
	specifity_vec = zeros(numClasses);
	VPP_vec = zeros(numClasses);
	VPN_vec = zeros(numClasses);
	F1_vec= zeros(numClasses);

	for class in 1:numClasses
		class_outputs = outputs[:,class]
		class_targets = targets[:,class]

		if length(class_outputs) ==0 # Si no hay patrones para esta clase???
			continue;
		end

		(acc,err,recall,specifity,VPP,VPN,F1,confusion_matrix) =
			confusionMatrix(class_outputs,class_targets);
		recall_vec[class] = recall;
		specifity_vec[class] = specifity;
		VPP_vec[class] = VPP;
		VPN_vec[class] = VPN;
		F1_vec[class] = F1;
	end


	# confusion_matrix = Matrix{Int}(undef,numClasses,numClasses);

	# for real in 1:numClasses
	# 	for prediction in 1:numClasses
	# 		real_targets = targets[:,real];
	# 		prediction_output = outputs[:,prediction]

	# 		# Producto escalar para contar
	# 		confusion_matrix[real,prediction] = real_targets' * prediction_output;  
	# 	end
	# end

	# En vez de hacer productos escalares, multiplicar las matrices
	# para tener la matriz de confusion
	
	confusion_matrix = targets' * outputs

	overall_recall = 0;
	overall_specifity = 0;
	overall_VPP = 0;
	overall_VPN = 0;
	overall_F1 = 0;
	if strat == macro_strat
		overall_recall = mean(recall_vec);
		overall_specifity = mean(specifity_vec);
		overall_VPP = mean(VPP_vec);
		overall_VPN = mean(VPN_vec);
		overall_F1= mean(F1_vec);
	elseif strat == weighted_strat
		classes_num_instances = sum(targets,dims=1)[:,1];
		weights = classes_num_instances/numInstances;

		overall_recall = sum(recall_vec .* weights);
		overall_specifity = sum(specifity_vec .* weights);
		overall_VPP = sum(VPP_vec .* weights);
		overall_VPN = sum(VPN_vec .* weights);
		overall_F1 = sum(F1_vec .* weights);
	else
		@assert false "Strat not supported"
	end


	acc = accuracy(targets,outputs);
	err = 1-acc;
	return (acc,err,overall_recall,overall_specifity,overall_VPP,
		overall_VPN,overall_F1,confusion_matrix);
end

function confusionMatrix(outputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2},strat::Strat)
	new_outputs = classifyOutputs(outputs);
	return confusionMatrix(new_outputs,targets,strat);
end

function confusionMatrix(outputs::AbstractArray{<:Any},targets::AbstractArray{<:Any},strat::Strat)
	@assert all(in.(outputs,(unique(targets),)))
	classes = unique(targets)

	new_outputs = oneHotEncoding(outputs,classes);
	new_targets = oneHotEncoding(targets,classes);
	return confusionMatrix(new_outputs,new_targets,strat);
end

function unoContraTodos()
	dataset = readdlm("iris.data",',');

	inputs = dataset[:,1:4];
	targets = dataset[:,5];
	inputs = convert(Array{Float32,2},inputs);
	targets = oneHotEncoding(targets);

	inputs = normalizeZeroMean(inputs);
	

	fit = trainDataset;

	numClasses = size(targets,2);
	numInstances = size(targets,1);
	outputs = Array{Float32,2}(undef,numInstances,numClasses);

	for numClass in 1:numClasses
		model = fit(inputs,targets[:,numClass]);
		outputs[:,numClass] .= model(inputs')';
	end

	outputs = softmax(outputs,dims=2);
	outputs = classifyOutputs(outputs);
end

##Cross-validation 
function crossvalidation(N::Int, k::Int)
	repetitions = Int(ceil(N/k)) # ceil no asegura que haya la misma cantidad de elementos en cada k
	subsets_indexes = repeat(1:1:k, repetitions)[1:N]
	shuffle!(subsets_indexes)

	return subsets_indexes
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int)
	return collect(Iterators.flatten(crossvalidation.(sum(targets, dims=1), k)))
	# el dot operation devuelve un array de arrays, 
	# que es engorroso para trabajar, por eso la conversión
	
	# Lo hice asumiendo que el array de targets está ordenado
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int)
	# onehot = oneHotEncoding(targets)
	# return crossvalidation(onehot, k)
	unique_elem = unique(targets)
	occurrences = [count(x -> x==n, targets) for n in unique_elem]
	return collect(Iterators.flatten(crossvalidation.(occurrences, k)))
end

function trainCrossValidation(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, kfolds::Int=10)
	Random.seed!(100)

	dataset_size = size(targets,1)
	train_idx = crossvalidation(targets, kfolds)

	# Solo lo hice para una métrica (la verdad es que no sé exactamente a qué métricas se refería, las de error de la RR.NN. u otras)
	train_err = zeros(kfolds)
	validation_err = zeros(kfolds)
	test_err = zeros(kfolds)
	
	for i = 1:kfolds
		# Habrá alguna manera más óptima de hacerlo
		k_idx = findall(x -> x==i, train_idx)
		not_k_idx = [j for j in 1:dataset_size if !(j in k_idx)]
		# No verefica que el ratio de conjunto de patrones sea mayor que 0 (length(not_k_idx) > 0)
		
		train_slice_idx, validation_slice_idx = holdOut(size(not_k_idx, 1), 0.3)
		train_slice_idx, validation_slice_idx  = not_k_idx[train_slice_idx], not_k_idx[validation_slice_idx]

		train_set, train_target_set = inputs[train_slice_idx, :], targets[train_slice_idx, :]
		test_set, test_target_set = inputs[k_idx, :], targets[k_idx, :]
		validation_set, validation_targets_set = inputs[validation_slice_idx, :], targets[validation_slice_idx, :]

		train = (train_set, train_target_set)
		test = (test_set, test_target_set)
		validation = (validation_set, validation_targets_set)

		train_params = calculateZeroMeanNormalizationParameters(train_set);
		inputs = normalizeZeroMean(inputs, train_params);

		(ann, k_train_err, k_validation_err, k_test_err) = trainRNA([12,4], train, maxEpochs=1000, 
			learningRate=0.01, validation=validation, test=test, maxEpochsVal=4);

		# Hago la media de los errores de cada iteración entrenamiento de la red, no sé si era eso realmente
		train_err[i] = k_train_err[end]
		validation_err[i] = k_validation_err[end]
		test_err[i] = k_test_err[end]
	end

	# Luego devuelvo la media de todos los k entrenamientos para el test set (es lo mínimo que pedía)
	return (mean(train_err), mean(test_err), mean(test_err))
	
end

# 6 4 3
l = [1,1,1,1,1,1,2,2,2,2,3,3,3]
k = crossvalidation(l, 3) # Equilibrado
shuffled_l = [2, 2, 1, 1, 3, 2, 2, 3, 1, 3, 1, 1, 1]
sk = crossvalidation(shuffled_l, 3)
unique.([k[1:6], k[7:10], k[11:13]])
unique.([sk[1:6], sk[7:10], sk[11:13]])

## Modificación de crossvalidation
function trainCrossValidation(inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}, kfolds::Int=10, )
	Random.seed!(100)

	dataset_size = size(targets,1)
	train_idx = crossvalidation(targets, kfolds)

	# Solo lo hice para una métrica (la verdad es que no sé exactamente a qué métricas se refería, las de error de la RR.NN. u otras)
	train_err = zeros(kfolds)
	validation_err = zeros(kfolds)
	test_err = zeros(kfolds)
	
	for i = 1:kfolds
		# Habrá alguna manera más óptima de hacerlo
		k_idx = findall(x -> x==i, train_idx)
		not_k_idx = [j for j in 1:dataset_size if !(j in k_idx)]
		# No verefica que el ratio de conjunto de patrones sea mayor que 0 (length(not_k_idx) > 0)
		
		train_slice_idx, validation_slice_idx = holdOut(size(not_k_idx, 1), 0.3)
		train_slice_idx, validation_slice_idx  = not_k_idx[train_slice_idx], not_k_idx[validation_slice_idx]

		train_set, train_target_set = inputs[train_slice_idx, :], targets[train_slice_idx, :]
		test_set, test_target_set = inputs[k_idx, :], targets[k_idx, :]
		validation_set, validation_targets_set = inputs[validation_slice_idx, :], targets[validation_slice_idx, :]

		train = (train_set, train_target_set)
		test = (test_set, test_target_set)
		validation = (validation_set, validation_targets_set)

		train_params = calculateZeroMeanNormalizationParameters(train_set);
		inputs = normalizeZeroMean(inputs, train_params);

		(ann, k_train_err, k_validation_err, k_test_err) = trainRNA([12,4], train, maxEpochs=1000, 
			learningRate=0.01, validation=validation, test=test, maxEpochsVal=4);

		# Hago la media de los errores de cada iteración entrenamiento de la red, no sé si era eso realmente
		train_err[i] = k_train_err[end]
		validation_err[i] = k_validation_err[end]
		test_err[i] = k_test_err[end]
	end

	# Luego devuelvo la media de todos los k entrenamientos para el test set (es lo mínimo que pedía)
	return (mean(train_err), mean(test_err), mean(test_err))
	
end


# No muy seguro de cómo interpretar el valor qualitativamente, es la media de los valores de crossentropy que da la RR.NN.
errs = trainCrossValidation(inputs, targets)

##Funcion para testear la confusion matrix
function trainDataset(inputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,2})
	
	dataset_size = size(targets,1)
	(train_idx,validation_idx,test_idx) = holdOut(dataset_size,0.3,0.3)
	
	train_inputs = inputs[train_idx,:];
	train_params = calculateZeroMeanNormalizationParameters(train_inputs);
	inputs = normalizeZeroMean(inputs,train_params);
	
	train = (inputs[train_idx,:],targets[train_idx,:]);
	validation = (inputs[validation_idx,:],targets[validation_idx,:]);
	test = (inputs[test_idx,:],targets[test_idx,:]);
	
	(ann,train_vector,validation_vector,test_vector) = trainRNA([12,4],train,maxEpochs=1000,learningRate=0.01,
		test=test,validation=validation,maxEpochsVal=4);
	return ann
end

function trainDataset(inputs::AbstractArray{<:Real,2},targets::AbstractArray{Bool,1})
 	new_targets = reshape(targets,length(targets),1);
	return trainDataset(inputs,new_targets);
end

dataset = readdlm("iris.data",',');

inputs = dataset[:,1:4];
targets = dataset[:,5];

@assert (size(inputs,1)==size(targets,1)) "Las matrices de entradas y salidas deseadas no tienen el mismo número de filas"



inputs = convert(Array{Float32,2},inputs);
targets = oneHotEncoding(targets);

dataset_size = size(targets,1)
(train_idx,validation_idx,test_idx) = holdOut(dataset_size,0.3,0.3)

train_inputs = inputs[train_idx,:];
train_params = calculateZeroMeanNormalizationParameters(train_inputs);
inputs = normalizeZeroMean(inputs,train_params);

train = (inputs[train_idx,:],targets[train_idx,:]);
validation = (inputs[validation_idx,:],targets[validation_idx,:]);
test = (inputs[test_idx,:],targets[test_idx,:]);

(ann,train_vector,validation_vector,test_vector) = trainRNA([12,4],train,maxEpochs=1000,learningRate=0.01,
	test=test,validation=validation,maxEpochsVal=4);

acc = accuracy(test[2],ann(test[1]')');

println(acc)

g = plot();
plot!(g,0:(length(train_vector)-1),train_vector,xaxis="Epoch",yaxis="Loss",color=:red,label="train");
plot!(g,0:(length(validation_vector)-1),validation_vector,xaxis="Epoch",yaxis="Loss",color=:blue,label="validation");
plot!(g,0:(length(test_vector)-1),test_vector,xaxis="Epoch",yaxis="Loss",color=:green,label="test");

display(g)

# savefig(g,"graph.svg")