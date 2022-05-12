using Statistics
using Flux
using Flux.Losses
using Flux: onehotbatch, onecold, params
using Random
using ScikitLearn

@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier


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
	# Quito el assert porque si se hace un holdOut, puede ser que no se distribuyan
	# los patrones de forma estratificada (Me hizo falta esto para la parte de redes convolucionales)
	# @assert all(in.(outputs,(unique(targets),)))
	# Combinar las clases de outputs y targets
	classes = unique([outputs targets])

	new_outputs = oneHotEncoding(outputs,classes);
	new_targets = oneHotEncoding(targets,classes);
	return confusionMatrix(new_outputs,new_targets,strat);
end

##Cross-validation 
function crossvalidation(N::Int, k::Int)
	repetitions = Int(ceil(N/k)) # ceil no asegura que haya la misma cantidad de elementos en cada k
	subsets_indexes = repeat(1:1:k, repetitions)[1:N]
	shuffle!(subsets_indexes)

	return subsets_indexes
end

function crossvalidation(targets::AbstractArray{Bool, 2}, k::Int)

	index_vector = Vector{Int}(undef,size(targets,1))
	classes = size(targets,2)

	if classes == 1
		class_targets = targets[:,1];
		class_num_elems = sum(class_targets);
		class_cross_val = crossvalidation(class_num_elems,k)
		class_indexes = findall( x-> x==1,class_targets);
		index_vector[class_indexes] = class_cross_val

		not_class_num_elems = size(targets,1) - sum(class_targets);
		not_class_cross_val = crossvalidation(not_class_num_elems,k)
		not_class_indexes = findall( x-> x==0,class_targets);
		index_vector[not_class_indexes] = not_class_cross_val

	else 
		for i = 1:classes
			class_targets = targets[:,i];
			num_elems = sum(class_targets);
			cross_val = crossvalidation(num_elems,k);
			class_indexes = findall(x->x==1,class_targets);
			index_vector[class_indexes] = cross_val;
		end
	end
	return index_vector
end

function crossvalidation(targets::AbstractArray{<:Any,1}, k::Int)
	onehot = oneHotEncoding(targets)
	return crossvalidation(onehot, k)
end


function repeatTrainRna(train_set::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},test_set::Tuple{AbstractArray{<:Real,2},AbstractArray{Bool,2}},parameters::Dict{Any,Any})

	(train_input_set,train_target_set)  = train_set;
	(test_input_set,test_target_set) = test_set;
		
	# HoldOut para RNA, lo hago fuera del bucle...
	(train_rna_idx,val_rna_idx) = holdOut(size(train_input_set,1),parameters["validation_ratio"])

	train = (train_input_set[train_rna_idx,:], train_target_set[train_rna_idx,:]);
	validation = (train_input_set[val_rna_idx,:], train_target_set[val_rna_idx,:]);

	executions = parameters["rna_executions"];
	acc_rna = zeros(executions)
	recall_rna = zeros(executions)
	specifity_rna = zeros(executions)
	f1_rna = zeros(executions)

	for j in 1:executions
		(ann, k_train_err, k_validation_err, k_test_err) = trainRNA(parameters["topology"], train, maxEpochs=parameters["max_epochs"], 
			learningRate=parameters["learning_rate"], validation=validation, test=test_set, maxEpochsVal=parameters["max_epochs_val"]);

		test_outputs = ann(test_input_set')'
		(acc_rna[j],_,recall_rna[j],specifity_rna[j],_,_,f1_rna[j],_) =
			confusionMatrix(test_outputs,test_target_set,macro_strat)
		
	end
	return (mean(acc_rna), mean(recall_rna),mean(specifity_rna),mean(f1_rna));
end

function trainConv(parameters::Dict{Any,Any},train_set::Tuple{AbstractArray{<:Real,3},AbstractArray{Bool,2}},
	test_set::Tuple{AbstractArray{<:Real,3},AbstractArray{Bool,2}})

	# ann = createRNA(topology,numInputs,numOutputs);

	

	ann = deepcopy(parameters["ann"])

	# Definimos la funcion de loss de forma similar a las prácticas de la asignatura
	loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
	# Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
	f1(batch) = confusionMatrix(onecold(ann(batch[1])),onecold(batch[2]),macro_strat)[7];

	# println("Ciclo 0: f1-score en el conjunto de entrenamiento: $(100 * f1(train_set))%");



	# Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
	opt = ADAM(0.01);

	train_loss_vector = Vector{Real}();
	test_loss_vector = Vector{Real}();

	push!(train_loss_vector,loss(train_set[1],train_set[2]));
	push!(test_loss_vector,loss(test_set[1],test_set[2]));
	batch_size = 128
	
	gruposIndicesBatch = Iterators.partition(1:size(train_set[1],3), batch_size);
	batch_train_set = [ (train_set[1][:,:,indicesBatch], train_set[2][:,indicesBatch]) for indicesBatch in gruposIndicesBatch];



	# println("Comenzando entrenamiento...")
	mejorF1 = -Inf;
	criterioFin = false;
	numCiclo = 0;
	numCicloUltimaMejora = 0;
	mejorModelo = nothing;

	while (!criterioFin)


		# Se entrena un ciclo
		Flux.train!(loss, params(ann), batch_train_set, opt);

		push!(train_loss_vector,loss(train_set[1],train_set[2]))
		push!(test_loss_vector,loss(test_set[1],test_set[2]));
		numCiclo += 1;

		# Se calcula la precision en el conjunto de entrenamiento:
		f1Entrenamiento = f1(train_set);
		println("Ciclo ", numCiclo, ": F1-Score en el conjunto de entrenamiento: ", 100*f1Entrenamiento, " %");

		# Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
		if (f1Entrenamiento > mejorF1)
			mejorF1 = f1Entrenamiento;
			f1Test = f1(test_set);
			println("Mejora en el conjunto de entrenamiento -> F1-Score en el conjunto de test: ", 100*f1Test, " %");
			mejorModelo = deepcopy(ann);
			numCicloUltimaMejora = numCiclo;
		end

		# Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
		if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
			opt.eta /= 10.0
			println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
			numCicloUltimaMejora = numCiclo;
		end

		# Criterios de parada:

		# Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
		if (f1Entrenamiento >= 0.999)
			# println("   Se para el entenamiento por haber llegado a un F1-Score de 99.9%")
			criterioFin = true;
		end

		# Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
		if (numCiclo - numCicloUltimaMejora >= 10)
			# println("   Se para el entrenamiento por no haber mejorado el F1-Score en el conjunto de entrenamiento durante 10 ciclos")
			criterioFin = true;
		end
	end
	return (mejorModelo, train_loss_vector, test_loss_vector)
end



function repeatTrainConv(train_set::Tuple{AbstractArray{<:Real,3},AbstractArray{Bool,2}},test_set::Tuple{AbstractArray{<:Real,3},AbstractArray{Bool,2}},parameters::Dict{Any,Any})

	(test_input_set,test_target_set) = test_set;

	executions = parameters["conv_executions"];
	acc_rna = zeros(executions)
	recall_rna = zeros(executions)
	specifity_rna = zeros(executions)
	f1_rna = zeros(executions)

	for j in 1:executions
		println("Execution: $(j)")
		(ann, train_vector, test_vector) = trainConv(parameters, train_set, test_set);\

		test_outputs = ann(test_input_set)
		(acc_rna[j],_,recall_rna[j],specifity_rna[j],_,_,f1_rna[j],_) =
			confusionMatrix(test_outputs',test_target_set',macro_strat)
	end
	return (mean(acc_rna), mean(recall_rna),mean(specifity_rna),mean(f1_rna));
end


function modelCrossValidation(model_symbol::Symbol,parameters::Dict{Any,Any},
	inputs::AbstractArray{<:Real,2}, targets::AbstractArray{Any,1}; kfolds::Int=10 )
	# Pasar los inputs normalizados!!

	@assert (in(model_symbol,[:ANN,:kNN,:DecisionTree,:SVM,:CONV])) "model not supported"

	f1_folds = zeros(kfolds)
	recall_folds = zeros(kfolds)
	specifity_folds = zeros(kfolds)
	acc_folds = zeros(kfolds)

	fold_vector = crossvalidation(targets, kfolds)
	if model_symbol == :ANN
		targets = oneHotEncoding(targets)
	end

	if model_symbol == :CONV
		# Transformar matriz de m x n en m x 1 x n, para poder aplicar las 
		# convoluciones
		inputs = reshape(inputs,(size(inputs,1),1,size(inputs,2)))
		targets = oneHotEncoding(targets)'
	end

	for i = 1:kfolds
		println("Fold: $(i)")
		test_fold_idx = findall(x -> x==i, fold_vector)
		train_fold_idx = findall(x -> x!=i, fold_vector)

		if model_symbol == :ANN
			(train_input_set,train_target_set)  = (inputs[train_fold_idx,:],targets[train_fold_idx,:]);
			(test_input_set,test_target_set) = (inputs[test_fold_idx,:], targets[test_fold_idx,:]);
			train = (train_input_set, train_target_set);
			test = (test_input_set, test_target_set);
			(acc,recall,specifity,f1) = repeatTrainRna(train,test,parameters);
		elseif model_symbol == :CONV
			
			train  = (inputs[:,:,train_fold_idx],targets[:,train_fold_idx]);
			test = (inputs[:,:,test_fold_idx], targets[:,test_fold_idx]);

			(acc,recall,specifity,f1) = repeatTrainConv(train,test,parameters);


		elseif in(model_symbol,[:kNN,:DecisionTree,:SVM])
			(train_input_set,train_target_set)  = (inputs[train_fold_idx,:],targets[train_fold_idx]);
			(test_input_set,test_target_set) = (inputs[test_fold_idx,:], targets[test_fold_idx]);

			if model_symbol == :SVM
				model = SVC(kernel=parameters["kernel"],degree=parameters["kernelDegree"],
					gamma=parameters["kernelGamma"],C=parameters["C"]);
			elseif model_symbol == :DecisionTree
				model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);

			elseif model_symbol == :kNN
				model = KNeighborsClassifier(parameters["k"]);
			end 
			fit!(model,train_input_set,train_target_set)
			test_outputs = predict(model,test_input_set);
			(acc,_,recall,specifity,_,_,f1,_) =
				confusionMatrix(test_outputs,test_target_set,macro_strat)
		end

		acc_folds[i] = acc;
		recall_folds[i] = recall;
		specifity_folds[i] = specifity;
		f1_folds[i] = f1;
	end

	return ((mean(acc_folds),std(acc_folds)), (mean(recall_folds),std(recall_folds)),
		(mean(specifity_folds),std(specifity_folds)), (mean(f1_folds),std(f1_folds)))
	
end
