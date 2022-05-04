using Plots
using DelimitedFiles
using Printf

include("src/scikit/p6.jl")


dataset = readdlm("dataset/aprox4.csv",',');

headers = dataset[1:1,1:end-1]
inputs = dataset[2:end,1:end-1];
targets = dataset[2:end,end];

inputs = convert(Array{Float32,2},inputs);

dataset_size = size(targets,1);
inputs = normalizeZeroMean(inputs);

folds = 10;

parameters = Dict();
parameters["topology"] = [256];
parameters["learning_rate"] = 0.01;
parameters["validation_ratio"] = 0.2;
parameters["rna_executions"] = 10;
parameters["max_epochs"] = 200;
parameters["max_epochs_val"] = 5;

parameters["kernel"] = "rbf";
parameters["kernelDegree"] = 3;
parameters["kernelGamma"] = 0.01;
parameters["C"] = 100;


model_symbol = :SVM

test_ratio = 0.2
train_idx,test_idx = holdOut(size(targets,1),test_ratio)
train_inputs,train_targets = inputs[train_idx,:],targets[train_idx]
test_inputs,test_targets = inputs[test_idx,:],targets[test_idx]


classes = unique(targets)


if model_symbol == :ANN	
	train_targets = oneHotEncoding(train_targets,classes)
	test_targets = oneHotEncoding(test_targets,classes)

	
	(train_rna_idx,val_rna_idx) = holdOut(size(train_inputs,1),parameters["validation_ratio"])
	train = (train_inputs[train_rna_idx,:], train_targets[train_rna_idx,:]);
	validation = (train_inputs[val_rna_idx,:], train_targets[val_rna_idx,:]);
	test = (test_inputs, test_targets);
	
	(ann, train_vector, validation_vector, test_vector) = trainRNA(parameters["topology"], train, maxEpochs=parameters["max_epochs"], 
		learningRate=parameters["learning_rate"], validation=validation, test=test, maxEpochsVal=parameters["max_epochs_val"]);

	test_outputs = ann(test_inputs')'

	plotlyjs();
	g = plot();
	plot!(g,0:(length(train_vector)-1),train_vector,xaxis="Epoch",yaxis="Loss",color=:red,label="train");
	plot!(g,0:(length(validation_vector)-1),validation_vector,xaxis="Epoch",yaxis="Loss",color=:blue,label="validation");
	plot!(g,0:(length(test_vector)-1),test_vector,xaxis="Epoch",yaxis="Loss",color=:green,label="test");

	# display(g);
	path = "best_rna_train.svg"
	display(g)
	println("ANN training plot saved to $(path)")

elseif in(model_symbol,[:kNN,:DecisionTree,:SVM])

	if model_symbol == :SVM
		model = SVC(kernel=parameters["kernel"],degree=parameters["kernelDegree"],
			gamma=parameters["kernelGamma"],C=parameters["C"]);
	elseif model_symbol == :DecisionTree
		model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);

	elseif model_symbol == :kNN
		model = KNeighborsClassifier(parameters["k"]);
	end 
	fit!(model,train_inputs,train_targets)
	test_outputs = predict(model,test_inputs);

	test_outputs = oneHotEncoding(test_outputs,classes)
	test_targets = oneHotEncoding(test_targets,classes)
end

(acc,_,recall,specifity,_,
		_,f1,matrix) =confusionMatrix(test_outputs,test_targets,macro_strat)

