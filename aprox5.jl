using Plots
using DelimitedFiles
using Printf

include("src/scikit/p6.jl")


Random.seed!(100)
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
headers = headers[selectedFeaturesIdx]

inputs = dataset[2:end,selectedFeaturesIdx];
targets = dataset[2:end,end];

inputs = convert(Array{Float32,2},inputs);

dataset_size = size(targets,1);

# Para calcular los parámetros de normalización
norm_params = calculateZeroMeanNormalizationParameters(inputs)
for i in 1:length(selectedFeatures)
	featidx = findall(f -> f == selectedFeatures[i],headers)[1]
	println("$(headers[featidx]) & \$$(norm_params[1][featidx])\$ & \$$(norm_params[2][featidx])\$ \\\\")
end

inputs = normalizeZeroMean(inputs);


folds = 10;

models = [:ANN,:SVM,:DecisionTree,:kNN]

models_parameters = Vector{Vector{Dict{Any,Any}}}(undef, length(models));

acc_models = Vector{Vector{Float64}}(undef,length(models));
acc_std_models = Vector{Vector{Float64}}(undef,length(models));

recall_models = Vector{Vector{Float64}}(undef,length(models));
recall_std_models = Vector{Vector{Float64}}(undef,length(models));

specifity_models = Vector{Vector{Float64}}(undef,length(models));
specifity_std_models = Vector{Vector{Float64}}(undef,length(models));

f1_models = Vector{Vector{Float64}}(undef,length(models));
f1_std_models = Vector{Vector{Float64}}(undef,length(models));

#RNA
topologies = [[16],[32],[64],[128],[256],[512],[64,64],[64,128]]
rna_parameters = Vector{Dict{Any,Any}}(undef, length(topologies));

for i in 1:length(topologies)
	topology_parameters = Dict();
	topology_parameters["topology"] = topologies[i];
	topology_parameters["learning_rate"] = 0.01;
	topology_parameters["validation_ratio"] = 0.2;
	topology_parameters["rna_executions"] = 10;
	topology_parameters["max_epochs"] = 200;
	topology_parameters["max_epochs_val"] = 5;
	rna_parameters[i] = topology_parameters;
end
models_parameters[1] = rna_parameters

#SVM
# El grado se ignora por los kernels que no sean polinomiales
# El gamma solo sirve para kernels poly,rbf y sigmoid
# (kernel,degree,gamma,C)
svm_configs = [
	("poly",3,1,1),
	("rbf",3,1,1),
	("sigmoid",3,1,1),
	
	("poly",3,10,0.1),
	("rbf",3,10,0.1),
	("rbf",3,0.001,100),

	("rbf",3,0.01,100),
	("poly",3,100,0.001),
]
svm_parameters = Vector{Dict{Any,Any}}(undef, length(svm_configs));

for i in 1:length(svm_configs)
	config_parameters = Dict();
	config_parameters["kernel"] = svm_configs[i][1];
	config_parameters["kernelDegree"] = svm_configs[i][2];
	config_parameters["kernelGamma"] = svm_configs[i][3];
	config_parameters["C"] = svm_configs[i][4];
	svm_parameters[i] = config_parameters;
end
models_parameters[2] = svm_parameters;

# DecisionTree
tree_configs = [32 64 128 256 512 1024]
tree_parameters = Vector{Dict{Any,Any}}(undef, length(tree_configs));

for i in 1:length(tree_configs)
	config_parameters = Dict();
	config_parameters["max_depth"] = tree_configs[i];
	tree_parameters[i] = config_parameters;
end
models_parameters[3] = tree_parameters;

#kNN
knn_configs = [2 3 7 15 31 63]
knn_parameters = Vector{Dict{Any,Any}}(undef, length(knn_configs));

for i in 1:length(knn_configs)
	config_parameters = Dict();
	config_parameters["k"] = knn_configs[i];
	knn_parameters[i] = config_parameters;
end
models_parameters[4] = knn_parameters;

# Models cross validation

for i in 1:length(models)
	configurations = models_parameters[i]
	num_configurations = length(configurations)

	acc_configurations = zeros(num_configurations)
	acc_std_configurations = zeros(num_configurations)

	recall_configurations = zeros(num_configurations)
	recall_std_configurations = zeros(num_configurations)

	specifity_configurations = zeros(num_configurations)
	specifity_std_configurations = zeros(num_configurations)

	f1_configurations = zeros(num_configurations)
	f1_std_configurations = zeros(num_configurations)

	for j in 1:num_configurations
		println("Training $(models[i]) (config $(j))")
		configuration_parameters = configurations[j]
		((acc_configurations[j],acc_std_configurations[j])), (recall_configurations[j],recall_std_configurations[j]),
			(specifity_configurations[j],specifity_std_configurations[j]), (f1_configurations[j],f1_std_configurations[j]) = modelCrossValidation(models[i],configuration_parameters,inputs,targets,kfolds=folds)
	end
	acc_models[i] = acc_configurations
	acc_std_models[i] = acc_std_configurations

	recall_models[i] = recall_configurations
	recall_std_models[i] = recall_std_configurations

	specifity_models[i] = specifity_configurations
	specifity_std_models[i] = specifity_std_configurations

	f1_models[i] = f1_configurations
	f1_std_models[i] = f1_std_configurations
end

# Resultados 
# for i in 1:length(models)

# 	println("--------------------------------")
# 	println("Model: $(models[i])")
# 	configurations = models_parameters[i]
# 	for j in 1:length(configurations)
# 		println("")
# 		configuration_parameters = configurations[j]
# 		for key in keys(configuration_parameters)
# 			println("$(key): $(configuration_parameters[key])")
# 		end
# 		println("Accuracy: $(round(acc_models[i][j],digits=5)) ± $(round(acc_std_models[i][j],digits=5))")
# 		println("F1-Score: $(round(f1_models[i][j],digits=5)) ± $(round(f1_std_models[i][j],digits=5))")
# 		println("")
# 	end
# 	println("--------------------------------")

# end

# Resultados para latex
for i in 1:length(models)

	if models[i] == :ANN
		config_name ="Arquitectura"
		caption="RNA"
	elseif models[i] == :SVM
		config_name ="(kernel, grado, gamma, C)"
		caption="SVM"
	elseif models[i] == :DecisionTree
		config_name="Altura máxima"
		caption="Árbol de decisión"
	elseif models[i] == :kNN
		config_name="K"
		caption="kNN"
	else
		config_name=""
	end

	println("----------------------------")
	println("$(models[i])\n")
	println("\\begin{table}[!ht]")
	println("\\caption{Resultados $(caption)}")
	println("\\centering")
	println("\t \\begin{tabular}{||c c c||} ")
	println("\t\t \\hline ")
	println("\t\t $(config_name) & F1-Score & Precisión  \\\\ [0.5ex]  ")
	println("\t\t \\hline\\hline")
	configurations = models_parameters[i]
	for j in 1:length(configurations)
		configuration_parameters = configurations[j]
		if models[i] == :ANN
			config = configuration_parameters["topology"]
		elseif models[i] == :SVM
			kernel = configuration_parameters["kernel"]
			kernelDegree = configuration_parameters["kernelDegree"]
			kernelGamma = configuration_parameters["kernelGamma"]
			C = configuration_parameters["C"]
			config="($(kernel), \$$(kernelDegree)\$, \$$(kernelGamma)\$, \$$(C)\$)"
		elseif models[i] == :DecisionTree
			max_depth = configuration_parameters["max_depth"]
			config="\$$(max_depth)\$"
		elseif models[i] == :kNN
			k = configuration_parameters["k"]
			config="\$$(k)\$"
		else
			config=""
		end
		println("\t\t $(config) & \$$(round(f1_models[i][j],digits=5)) \\pm $(round(f1_std_models[i][j],digits=5))\$ & \$$(round(acc_models[i][j],digits=5)) \\pm $(round(acc_std_models[i][j],digits=5))\$ \\\\")
		println("\t\t \\hline ")
	end
	println("\t \\end{tabular}")
	println("\\label{Tab:$(models[i])} ")
	println("\\end{table} ")

	println("\n----------------------------")
end

best_config=(1,1);
best_mean_f1 = f1_models[1][1];
best_std_f1 = f1_std_models[1][1];
for i in 1:length(models)
	model_configurations = models_parameters[i]
	for j in 1:length(model_configurations)
		if f1_models[i][j] == best_mean_f1
			# A igual f1, elijo la que tenga menos std
			if f1_std_models[i][j] < best_std_f1
				global best_std_f1 = f1_std_models[i][j]
				global best_config = (i,j)
			end
		elseif f1_models[i][j] > best_mean_f1
			global best_mean_f1 = f1_models[i][j]
			global best_config = (i,j)
		end
	end
end
(best_model,best_model_config) = best_config
println("Best configuration\n")
println("Model: $(models[best_model])")
model_configuration = models_parameters[best_model][best_model_config]
println("Hyperparameters:")
for key in keys(model_configuration)
	println("\t$(key): $(model_configuration[key])")
end
println("Accuracy: $(round(acc_models[best_model][best_model_config],digits=5)) ± $(round(acc_std_models[best_model][best_model_config],digits=5))")
println("F1-Score: $(round(f1_models[best_model][best_model_config],digits=5)) ± $(round(f1_std_models[best_model][best_model_config],digits=5))")


# Resultados para el mejor modelo

# Un 20% de las instancias son para generar la matriz de confusión (test)

test_ratio = 0.2
train_idx,test_idx = holdOut(size(targets,1),test_ratio)
train_inputs,train_targets = inputs[train_idx,:],targets[train_idx]
test_inputs,test_targets = inputs[test_idx,:],targets[test_idx]


classes = unique(targets)

parameters = models_parameters[best_model][best_model_config]

if models[best_model] == :ANN	
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
	savefig(g,path)
	println("ANN training plot saved to $(path)")

elseif in(models[best_model],[:kNN,:DecisionTree,:SVM])

	if models[best_model] == :SVM
		model = SVC(kernel=parameters["kernel"],degree=parameters["kernelDegree"],
			gamma=parameters["kernelGamma"],C=parameters["C"]);
	elseif models[best_model] == :DecisionTree
		model = DecisionTreeClassifier(max_depth=parameters["max_depth"], random_state=1);

	elseif models[best_model] == :kNN
		model = KNeighborsClassifier(parameters["k"]);
	end 
	fit!(model,train_inputs,train_targets)
	test_outputs = predict(model,test_inputs);

	test_outputs = oneHotEncoding(test_outputs,classes)
	test_targets = oneHotEncoding(test_targets,classes)
end

println("------------------------------")
println("Results using whole dataset for best model with $(test_ratio) test ratio:\n")

(acc,_,recall,specifity,_,
		_,f1,matrix) =confusionMatrix(test_outputs,test_targets,macro_strat)


# Print confusion matrix
@printf "%-3s " ""
for i in 1:length(classes)
	@printf "%-3s " classes[i]
end
@printf "\n"
for i in 1:size(matrix,1)
	@printf "%-3s " classes[i]
	for j in 1:size(matrix,2)
		@printf "%-3d " matrix[i,j]
	end
	@printf "\n"
end

println("Accuracy: $(round(acc,digits=5))")
println("F1-Score: $(round(f1,digits=5))")
println("Recall: $(round(recall,digits=5))")
println("Specifity: $(round(specifity,digits=5))")


