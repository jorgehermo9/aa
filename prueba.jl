using Plots
using DelimitedFiles
using Printf

include("src/scikit/p6.jl")


dataset = readdlm("dataset/aprox4.csv",',');

headers = dataset[1,1:end-1]

# Selección de características
# Top 70 características con mrmr-enhanced. Discretización
# del dataset con 10 bins utilizando estrategia de cuantiles.
selectedFeatures = [
	"abs_max_freq",
	"max_freq19",
	"max_freq17",
	"max_freq14",
	"max_freq13",
	"max_freq21",
	"max_freq12",
	"max_freq27",
	"max_freq11",
	"max_freq20",
	"max_freq16",
	"max_freq23",
	"max_freq7",
	"max_freq24",
	"max_freq10",
	"max_freq15",
	"max_freq22",
	"max_freq9",
	"max_freq18",
	"max_freq8",
	"max_freq25",
	"max_freq26",
	"max_freq28",
	"zero_crossing",
	"max_freq5",
	"max_freq6",
	"max_freq29",
	"max_freq30",
	"max8",
	"max_freq4",
	"max6",
	"std8",
	"std6",
	"abs_max",
	"std7",
	"max11",
	"max9",
	"max12",
	"max10",
	"max14",
	"max7",
	"max13",
	"max_freq2",
	"max5",
	"std10",
	"max17",
	"max15",
	"std9",
	"max_freq3",
	"max16",
	"max4",
	"m8",
	"std11",
	"E",
	"m6",
	"std14",
	"std13",
	"std5",
	"std12",
	"std4",
	"m7",
	"std15",
	"max18",
	"max_freq1",
	"max19",
	"std16",
	"std17",
	"m4",
	"max21",
	"m10",
]

selectedFeaturesIdx = findall(x -> in(x,selectedFeatures),headers)
display(selectedFeaturesIdx)
inputs = dataset[2:end,selectedFeaturesIdx];
targets = dataset[2:end,end];

inputs = convert(Array{Float32,2},inputs);

dataset_size = size(targets,1);
inputs = normalizeZeroMean(inputs);

folds = 10;

parameters = Dict();
parameters["topology"] = [64];
parameters["learning_rate"] = 0.01;
parameters["validation_ratio"] = 0.2;
parameters["rna_executions"] = 1;
parameters["max_epochs"] = 300;
parameters["max_epochs_val"] = 5;

parameters["kernel"] = "rbf";
parameters["kernelDegree"] = 3;
parameters["kernelGamma"] = 0.001;
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

